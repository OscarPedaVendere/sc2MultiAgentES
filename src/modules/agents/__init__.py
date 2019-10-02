REGISTRY = {}

from .rnn_agent import RNNAgent
from .es_rnn_agent import EsRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["es_rnn"] = EsRNNAgent