from .q_learner import QLearner
from .coma_learner import COMALearner
from .es_learner import ESLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["es_learner"] = ESLearner
