# Primero vamos a loguear de un dataset

from autogoal.sampling import UnormalizedWeightParam
from autogoal.logging import logger
from autogoal.ml import AutoML
from autogoal.kb import MatrixContinuousDense, VectorCategorical, Supervised
from autogoal.experimental.metalearning import DatasetFeatureLogger, MetalearningModel
from autogoal.search import RichLogger

from autogoal.datasets import cars

X,y = cars.load()

automl = AutoML(
    input=(MatrixContinuousDense, Supervised[VectorCategorical]),
    output=VectorCategorical,
    search_timeout=600,
    evaluation_timeout=5,
)

# Crear el logger, necesita el dataset para extraer features del problema
metalogger = DatasetFeatureLogger(X, y)

# automl.fit(X, y, logger=metalogger)

"""
>>> automl.fit(X, y, logger=metalogger)
"""

# Ahora creamos un MetalearningModel que aprenderá un modelo de sampling de este dataset
from autogoal.experimental.metalearning import MetalearningModel

model = MetalearningModel().build_model()

# Veamos que aprendió el modelo. Aquellos algoritmos que mejor resultados hayan dado
# deben tener pesos mayores:

for k,v in model.items():
    if isinstance(v, UnormalizedWeightParam):
        print(k, v.value)

# Ahora queremos ver con este modelo, se obtienen mejores resultados al hacer AutoML.
# Para eso se le pasa al AutoML como modelo inicial (`initial_model`), que la clase `AutoML`
# pasará como parámetro al algoritmo de optimización (por defecto `PESearch`).

automl = AutoML(
    input=(MatrixContinuousDense, Supervised[VectorCategorical]),
    output=VectorCategorical,
    search_timeout=600,
    evaluation_timeout=5,
    initial_model=model,
)

# Veamos que tal funciona

automl.fit(X, y, logger=RichLogger())