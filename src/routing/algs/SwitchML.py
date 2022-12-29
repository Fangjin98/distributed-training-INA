from typing import List
from routing.algs.ATP import ATP
from routing.utils.TopoGenerator import TopoGenerator


class SwitchML(ATP):
    def __init__(self, topo: TopoGenerator,test_set, **kwargs) -> None:
        super().__init__(topo,test_set,**kwargs)
        
    def run(self):
        tor_switch_set=self._get_tor_switch()
        candidate_paths=self._get_candidate_path()
        
        aggregation_node=dict()
        routing_paths=[]
        
        for w in self.flatten_worker_set:
            aggregation_node[w]=tor_switch_set[w]
            routing_paths.append(
                self._choose_path(candidate_paths[w][tor_switch_set[w]])
            )
        
        for s in self.switch_set:
            routing_paths.append(
                self._choose_path(candidate_paths[s][self.ps])
            )
        
        return aggregation_node, routing_paths
    
    