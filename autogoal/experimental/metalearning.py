from os import P_PGID, pipe
import uuid
import abc
import functools
import warnings
import json
import collections
import numpy as np
from pathlib import Path

from typing import Dict, List
from autogoal.search import Logger, SearchAlgorithm
from autogoal.sampling import ModelSampler, MeanDevParam, UnormalizedWeightParam, WeightParam, DistributionParam, ModelParam
from autogoal.utils import nice_repr

# from sklearn.feature_extraction import DictVectorizer


class DatasetFeatureLogger(Logger):
    def __init__(
        self,
        X,
        y=None,
        extractor=None,
        output_file="metalearning.json",
        problem_features=None,
        environment_features=None,
    ):
        self.extractor = extractor or DatasetFeatureExtractor()
        self.X = X
        self.y = y
        self.run_id = str(uuid.uuid4())
        self.output_file = output_file
        self.problem_features = problem_features or {}
        self.environment_features = environment_features or {}

    def begin(self, generations, pop_size, search_algorithm: SearchAlgorithm):
        self.dataset_features_ = self.extractor.extract_features(self.X, self.y)

        # TODO: Añadir a environment features cosas como el CPU y RAM disponible cuando se ejecutó el experimento.
        # Además en cada pipeline añadir la RAM usada y el tiempo de cómputo que estuvo evaluando.
        self.environment_features.update(
            evaluation_timeout=search_algorithm._evaluation_timeout,
            memory_limit=search_algorithm._memory_limit,
            search_timeout=search_algorithm._search_timeout,
            generations=generations,
            pop_size=pop_size,
        )

    def eval_solution(self, solution, fitness):
        if not hasattr(solution, "sampler_"):
            raise ("Cannot log if the underlying algorithm is not PESearch")

        sampler = solution.sampler_

        features = {k: v for k, v in sampler._updates.items() if isinstance(k, str)}
        feature_types = {k: v.__class__.__name__ for k, v in sampler._model.items() if k in features}

        info = SolutionInfo(
            uuid=self.run_id,
            fitness=fitness,
            problem_features=dict(self.dataset_features_, **self.problem_features),
            environment_features=dict(self.environment_features),
            pipeline_features=features,
            feature_types=feature_types,
        ).to_dict()

        with open(self.output_file, "a") as fp:
            fp.write(json.dumps(info) + "\n")


class DatasetFeatureExtractor:
    def __init__(self, features_extractors=None):
        self.feature_extractors = list(features_extractors or _EXTRACTORS)

    def extract_features(self, X, y=None):
        features = {}

        for extractor in self.feature_extractors:
            features.update(**extractor(X, y))

        return features


_EXTRACTORS = []


def feature_extractor(func):
    @functools.wraps(func)
    def wrapper(X, y=None):
        try:
            result = func(X, y)
        except:
            result = None
            # raise

        return {func.__name__: result}

    _EXTRACTORS.append(wrapper)
    return wrapper


# Feature extractor methods


@feature_extractor
def is_supervised(X, y=None):
    return y is not None


@feature_extractor
def dimensionality(X, y=None):
    d = 1

    for di in X.shape[1:]:

        d *= di

    return d


@feature_extractor
def training_examples(X, y=None):
    try:
        return X.shape[0]
    except:
        return len(X)


@feature_extractor
def has_numeric_features(X, y=None):
    return any([xi for xi in X[0] if isinstance(xi, (float, int))])


@feature_extractor
def numeric_variance(X, y=None):
    return X.std()


@feature_extractor
def average_number_of_words(X, y=None):
    return sum(len(sentence.split(" ")) for sentence in X) / len(X)


@feature_extractor
def has_text_features(X, y=None):
    return isinstance(X[0], str)


@nice_repr
class SolutionInfo:
    def __init__(
        self,
        uuid: str,
        problem_features: dict,
        pipeline_features: dict,
        environment_features: dict,
        feature_types: dict,
        fitness: float,
    ):
        self.problem_features = problem_features
        self.pipeline_features = pipeline_features
        self.environment_features = environment_features
        self.fitness = fitness
        self.feature_types = feature_types
        self.uuid = uuid

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(d):
        return SolutionInfo(**d)


class MetalearningModel:
    """
    Crea un ModelSampler que se inicializa con las distribuciones aprendidas del metalearning.

    A partir de los ejemplos en `db`, se re-definen las distribuciones asociadas a cada una
    de las producciones vistas, de forma tal que se asemejen a los mejores pipelines generados.
    """

    DISTRIBUTION_MAP = {cls.__name__: cls for cls in [WeightParam, UnormalizedWeightParam, DistributionParam, MeanDevParam]}

    def __init__(self, db="metalearning.json") -> None:
        self.db = db

    def build_model(self):
        examples: List[SolutionInfo] = []

        # Primero se cargan todos los ejemplos de metalearning

        with open(self.db) as fp:
            for line in fp:
                examples.append(SolutionInfo(**json.loads(line)))

        # Vamos a computar el mejor fitness por cada ejecución diferente
        # para normalizar cada pipeline evaluado en función de esa ejecución.
        # A cada ejecución además le vamos a computar su similaridad con el problema
        # actual.
        # Cada pipeline entonces tendrá un fitness normalizado con respecto a la ejecución
        # donde se evaluó y ponderado por la similaridad con el problema actual.

        best_fitness = {}
        similarity = {}

        for pipeline in examples:
            best_fitness[pipeline.uuid] = pipeline.fitness

            if pipeline.uuid not in similarity:
                similarity[pipeline.uuid] = self._get_problem_similarity(pipeline.problem_features)

        for pipeline in examples:
            bf = best_fitness[pipeline.uuid]

            if bf > 0:
                pipeline.fitness = pipeline.fitness * similarity[pipeline.uuid] / bf

        # Por cada producción usada en cada pipeline, vamos a ponderar esa distribución
        # en función de los pipelines que se la usan. Para eso vamos a crear un índice por
        # las distribuciones almacenando el valor que se sampleó y el fitness que se obtuvo.

        distribution_types: Dict[str, ModelParam] = {}
        distribution_values = collections.defaultdict(list)

        for pipeline in examples:
            for k,t in pipeline.feature_types.items():
                distribution_types[k] = self.DISTRIBUTION_MAP[t]

            for d,values in pipeline.pipeline_features.items():
                for v in values:
                    distribution_values[d].append((v, pipeline.fitness))

        # Finalmente cada clase de distribución sabe como construirse a partir de un
        # conjunto de samples ponderados (método `ModelParams.build`).

        distributions = {
            k: cls.build(distribution_values[k]) for k,cls in distribution_types.items()
        }

        return distributions


    def _get_problem_similarity(self, problem_features: dict):
        # TODO: Calcular similar con el problema actual
        return 1


# class LearnerMedia:
#     def __init__(self, problem, solutions: List[SolutionInfo], beta=1):
#         self.solutions = solutions
#         self.problem = problem
#         self.beta = beta

#     def initialize(self):
#         raise NotImplementedError("We need to refactor to not depend on DictVectorizer")

#         self.best_fitness = collections.defaultdict(lambda: 0)
#         self.all_features = {}

#         for i in self.solutions:
#             self.best_fitness[i.uuid] = max(self.best_fitness[i.uuid], i.fitness)

#             for feature in i.pipeline_features:
#                 self.all_features[feature] = None

#         self.vect = DictVectorizer(sparse=False)
#         self.vect.fit([self.problem])

#         self.weights_solution = self.calculate_weight_examples(self.solutions)

#     def compute_all_features(self):
#         self.initialize()

#         for feature in list(self.all_features):
#             self.all_features[feature] = self.compute_feature(feature)
#             print(feature, "=", self.all_features[feature])

#     def compute_feature(self, feature):
#         """Select for training all solutions where is used the especific feature.

#         Predict the media of the parameter value.
#         """
#         # find the relevant solutions, that contain the production to predict
#         important_solutions = []
#         feature_prototype = None

#         for i, w in zip(self.solutions, self.weights_solution):
#             if feature in i.pipeline_features:
#                 for value in i.pipeline_features[feature]:
#                     important_solutions.append((value, w))

#                 if feature_prototype is None:
#                     feature_prototype = eval(
#                         i.feature_types[feature], sampling.__dict__, {}
#                     )

#         if feature_prototype is None:
#             return None

#         return feature_prototype.weighted(important_solutions)

#     def calculate_weight_examples(self, solutions: List[SolutionInfo]):
#         """Calcule a weight of each example considering the fitness and the similariti with the
#         actual problem
#         """
#         # met = fitness * (similarity)^beta
#         # métrica utilizada en active learning para combinar informativeness with representativeness

#         weights = []

#         for info in solutions:
#             # normalize fitness
#             info.fitness = self.normalize_fitness(info)

#             if info.fitness == 0:
#                 continue

#             # calculate similarity
#             sim = self.similarity_cosine(info.problem_features)
#             # calculate metric for weight
#             weights.append(info.fitness * (sim) ** self.beta)

#         return weights

#     def normalize_fitness(self, info: SolutionInfo):
#         """Normalize the fitness with respect to the best solution in the problem where that solution is evaluated
#         """
#         return info.fitness / self.best_fitness[info.uuid]

#     def similarity_cosine(self, other_problem):
#         """Caculate the cosine similarity for a particular solution problem(other problem)
#         and the problem analizing
#         """
#         x = self.vect.transform(other_problem)[0]
#         y = self.vect.transform(self.problem)[0]

#         return np.dot(x, y) / (np.dot(x, x) ** 0.5 * np.dot(y, y) ** 0.5)

#     def similarity_learning(self, other_problem):
#         """ Implementar una espicie de encoding para los feature de los problemas
#         """
#         raise NotImplementedError()
