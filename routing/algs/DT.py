from routing.algs.BasicAlg import BasicAlg
from routing.utils.TopoGenerator import TopoGenerator


class DT(BasicAlg):
    def __init__(self, topo: TopoGenerator) -> None:
        super().__init__(topo)

    def run(self, test_set, comp, band):
        ps = test_set[0]
        worker_set = test_set[1]

        flatten_worker_set = []
        for ww in worker_set:
            for w in ww:
                flatten_worker_set.append(w)

        aggregation_node = dict()
        rate = dict()

        max_rate = band / len(flatten_worker_set)

        for w in flatten_worker_set:
            aggregation_node[w] = ps
            rate[w] = max_rate

        return aggregation_node, rate
