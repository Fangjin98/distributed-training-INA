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
    
    def cal_train_time(self,aggregation_points, t, comp, band, iteration=100) -> float:
        train_time=0


class SwitchMLDataAnalyzer(DataAnalyzer):
    def __init__(self, title: String, topo: TopoGenerator) -> None:
        super().__init__(title, topo)
    
    def cal_throughput(self, ps, worker_set, switch_set, aggregation_points, t) -> float:
        return super().cal_throughput(ps, worker_set, switch_set, aggregation_points, t)