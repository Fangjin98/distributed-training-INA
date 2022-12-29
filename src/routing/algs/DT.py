from routing.algs.BasicAlg import BasicAlg
from routing.utils.TopoGenerator import TopoGenerator


class DT(BasicAlg):
    def __init__(self, topo: TopoGenerator,test_set, **kwargs) -> None:
        super().__init__(topo)
        
        self.ps=test_set[0]
        self.worker_set=test_set[1]
        self.switch_set=test_set[2]
        
        self.flatten_worker_set=[]
        for ww in self.worker_set:
            for w in ww:
                self.flatten_worker_set.append(w)
        
    def run(self):
        aggregation_node=dict()
        routing_paths=[]
        
        for w in self.flatten_worker_set:
            aggregation_node[w]=self.ps
            routing_paths.append(
                self.topo.get_shortest_path(w,self.ps)[0]
            )
        
        return aggregation_node, routing_paths
    
    