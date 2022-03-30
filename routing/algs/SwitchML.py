from routing.algs.ATP import ATP
from routing.utils.TopoGenerator import TopoGenerator


class SwitchML(ATP):
    def __init__(self, topo: TopoGenerator) -> None:
        super().__init__(topo)

    def run(self, test_set, comp, band, **kwargs):
        ps = test_set[0]
        worker_set = test_set[1]
        switch_set = test_set[2]
        flatten_worker_set = []

        for ww in worker_set:
            for w in ww:
                flatten_worker_set.append(w)

        tor_switch_set = self._get_tor_switch(ps, worker_set, switch_set)

        aggregation_node = dict()
        rate = dict()
        count = {s: 0 for s in switch_set}
        max_rate = band / len(switch_set)

        for w in flatten_worker_set:
            aggregation_node[w] = tor_switch_set[w]
            count[tor_switch_set[w]] += 1

        for index, s in enumerate(switch_set):
            tmp_rate = max(max_rate, comp[index] / count)
            for w in flatten_worker_set:
                if aggregation_node[w] == s:
                    rate[w] = tmp_rate

        return aggregation_node, rate
