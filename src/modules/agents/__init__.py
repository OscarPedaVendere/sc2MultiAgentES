REGISTRY = {}

from .rnn_agent import RNNAgent
from .es_rnn import ESRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["es_rnn"] = ESRNNAgent