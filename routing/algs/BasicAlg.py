from abc import ABC, abstractmethod
from routing.utils.TopoGenerator import TopoGenerator


class BasicAlg(ABC):
    def __init__(self, topo: TopoGenerator) -> None:
        self.topo=topo

    @abstractmethod
    def run(self):
        pass
                    