from collections import defaultdict
from email.policy import default
import random
from typing import List
from routing.algs.BasicAlg import BasicAlg
from routing.utils.TopoGenerator import TopoGenerator


class ATP(BasicAlg):
    def __init__(self, topo: TopoGenerator) -> None:
        super().__init__(topo)

    def run(self, test_set, comp, band, **kwargs):
        ps = test_set[0]
        worker_set = test_set[1]
        switch_set = test_set[2]
        flatten_worker_set = []
        rate = kwargs['sending_rate']

        for ww in worker_set:
            for w in ww:
                flatten_worker_set.append(w)

        tor_switch_set = self._get_tor_switch(ps, worker_set, switch_set)
        capacity_dict = {
            switch: val for switch, val in zip(switch_set, comp)
        }
        aggregation_node = dict()

        for index, w in enumerate(flatten_worker_set):
            if capacity_dict[tor_switch_set[w]] >= rate[index]:  # aggregate on tier 0
                capacity_dict[tor_switch_set[w]] -= rate[index]
                aggregation_node[w] = tor_switch_set[w]
            elif capacity_dict[tor_switch_set[ps]] >= rate[index]:  # aggregate on tier 1
                capacity_dict[tor_switch_set[ps]] -= rate[index]
                aggregation_node[w] = tor_switch_set[ps]
            else:  # directly aggregate on ps
                aggregation_node[w] = ps

        return aggregation_node, rate

    @staticmethod
    def _get_tor_switch(ps, worker_set, switch_set):
        tor_switch = defaultdict(dict)

        for index, ww in enumerate(worker_set):
            for w in ww:
                tor_switch[w] = switch_set[index]

        tor_switch[ps] = switch_set[0]

        return tor_switch
