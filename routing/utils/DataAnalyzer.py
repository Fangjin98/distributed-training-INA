from tokenize import String
from typing import List

from routing.utils.TopoGenerator import TopoGenerator


class DataAnalyzer:
    def __init__(self, title: String, topo: TopoGenerator) -> None:
        self.title=title
        self.topo=topo
    
    def cal_throughput(self,ps, worker_set, switch_set, aggregation_points, t) -> float:
        throughput=0.0
        for w in worker_set:
            p=aggregation_points[w]
            if p in switch_set:
                throughput+=t
        return throughput
    
    def cal_train_time(self,worker_set, switch_set, aggregation_points,routing_paths, t, comp, ps_comp, band, iteration=100) -> float:
        train_time=0
        aggregated_switch=[]
        # computation time
        for w in worker_set:
            n=aggregation_points[w]
            if n in switch_set:
                train_time+=(t/comp)
                if n not in aggregated_switch:
                    aggregated_switch.append(n)
            else:
                train_time+=(t/ps_comp)
        train_time+=(t*len(aggregated_switch)/ps_comp)
        # transfer time
        for p in routing_paths:
            for l in p.link_list:
                train_time+=(t/band)
        
        return train_time*iteration


class SwitchMLDataAnalyzer(DataAnalyzer):
    def __init__(self, title: String, topo: TopoGenerator) -> None:
        super().__init__(title, topo)
