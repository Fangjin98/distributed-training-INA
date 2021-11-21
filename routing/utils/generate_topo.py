import json
from collections import defaultdict


class Topo(object):
    def __init__(self, topo=defaultdict(list)):
        self.topo = topo

    def add_edge(self, src, dst, weight):
        self.topo[src].append({dst: weight})

    def add_edges(self, edge_list):
        for e in edge_list:
            self.add_edge(e[0], e[1], e[2])

    def remove_edge(self, src, dst):
        if self.topo[src]:
            for node in self.topo[dst]:
                if node.keys()[0] is dst:
                    self.topo[src].remove(node)
                    break


def generate_json(topo, topo_file):
    json_str = json.dumps(topo, indent=4)
    with open(topo_file, 'w') as json_file:
        json_file.write(json_str)


def topo_25_worker_4_switch(host_set, switch_set):
    topo = Topo()
    topo.add_edges([('s1', 's2', 1), ('s1', 's3', 1), ('s1', 's4', 1)])
    for index, s in enumerate(switch_set[1:]):
        topo.add_edge(s, 's1', 1)
        for h in host_set[8 * index:8 * index + 8]:
            topo.add_edge(s, h, 1)
    for i in range(3):
        for h in host_set[8 * i:8 * i + 8]:
            topo.add_edge(h, switch_set[i + 1], 1)

    topo.add_edge('s4', 'h25', 1)
    topo.add_edge('h25', 's4', 1)
    return topo


if __name__ == '__main__':
    my_topo = topo_25_worker_4_switch(host_set=['h' + str(i) for i in range(1, 26)],
                                      switch_set=['s1', 's2', 's3', 's4'])
    print(my_topo.topo)
    # generate_json(my_topo.topo, '../data/topo/topo.json')
