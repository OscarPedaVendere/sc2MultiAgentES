REGISTRY = {}

from .basic_controller import BasicMAC
from .es_controller import EsMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["es_mac"] = EsMAC