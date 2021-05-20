# Primero vamos a loguear de un dataset

from autogoal.ml import AutoML
from autogoal.kb import MatrixContinuousDense, VectorCategorical, Supervised
from autogoal.experimental.metalearning import DatasetFeatureLogger
from autogoal.search import RichLogger

from autogoal.datasets import cars

X,y = cars.load()

automl = AutoML(
    input=(MatrixContinuousDense, Supervised[VectorCategorical]),
    output=VectorCategorical,
    search_timeout=60,
    evaluation_timeout=5,
)

# Crear el logger, necesita el dataset para extraer features del problema
metalogger = DatasetFeatureLogger(X, y)
automl.fit(X, y, logger=[metalogger, RichLogger()])