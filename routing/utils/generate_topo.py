import json
from collections import defaultdict


class Topo(object):
    def __init__(self, topo=defaultdict(list)):
        self.topo = topo

    def add_edge(self, src, dst, weight):
        self.topo[src].append({dst: weight})
        self.topo[dst].append({src: weight})

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


def topo_9_worker_4_switch():
    topo = Topo(defaultdict(list))
    topo.add_edges([('s1', 's3', 1), ('s1', 's4', 1), ('s2', 's4', 1), ('s2', 's3', 1)])
    topo.add_edges([('s1', 'h1', 1), ('s1', 'h2', 1), ('s1', 'h3', 1), ('s1', 'h4', 1), ('s2', 'h5', 1)])
    topo.add_edges([('s2', 'h6', 1), ('s2', 'h7', 1), ('s2', 'h8', 1), ('s2', 'h9', 1)])
    return topo


def topo_7_worker_4_switch():
    topo = Topo(defaultdict(list))
    topo.add_edges([('s1', 's3', 1), ('s1', 's4', 1), ('s2', 's4', 1), ('s2', 's3', 1)])
    topo.add_edges([('s1', 'h1', 1), ('s1', 'h2', 1), ('s1', 'h3', 1), ('s2', 'h4', 1)])
    topo.add_edges([('s2', 'h5', 1), ('s2', 'h6', 1), ('s2', 'h7', 1)])
    return topo


def topo_11_worker_4_switch():
    topo = Topo(defaultdict(list))
    topo.add_edges([('s1', 's3', 1), ('s1', 's4', 1), ('s2', 's4', 1), ('s2', 's3', 1)])
    topo.add_edges(
        [('s1', 'h1', 1), ('s1', 'h2', 1), ('s1', 'h3', 1), ('s1', 'h4', 1), ('s1', 'h5', 1), ('s2', 'h6', 1)])
    topo.add_edges([('s2', 'h7', 1), ('s2', 'h8', 1), ('s2', 'h9', 1), ('s2', 'h10', 1), ('s2', 'h11', 1)])
    return topo


def topo_13_worker_4_switch():
    topo = Topo(defaultdict(list))
    topo.add_edges([('s1', 's3', 1), ('s1', 's4', 1), ('s2', 's4', 1), ('s2', 's3', 1)])
    topo.add_edges(
        [('s1', 'h1', 1), ('s1', 'h2', 1), ('s1', 'h3', 1), ('s1', 'h4', 1), ('s1', 'h5', 1), ('s1', 'h6', 1),
         ('s2', 'h7', 1)])
    topo.add_edges(
        [('s2', 'h8', 1), ('s2', 'h9', 1), ('s2', 'h10', 1), ('s2', 'h11', 1), ('s2', 'h12', 1), ('s2', 'h13', 1)])
    return topo


def topo_15_worker_4_switch():
    topo = Topo(defaultdict(list))
    topo.add_edges([('s1', 's3', 1), ('s1', 's4', 1), ('s2', 's4', 1), ('s2', 's3', 1)])
    topo.add_edges(
        [('s1', 'h1', 1), ('s1', 'h2', 1), ('s1', 'h3', 1), ('s1', 'h4', 1), ('s1', 'h5', 1), ('s1', 'h6', 1),
         ('s1', 'h7', 1), ('s2', 'h8', 1)])
    topo.add_edges(
        [('s2', 'h9', 1), ('s2', 'h10', 1), ('s2', 'h11', 1), ('s2', 'h12', 1), ('s2', 'h13', 1), ('s2', 'h14', 1),
         ('s2', 'h15', 1)])
    return topo


if __name__ == '__main__':
    my_topo = topo_7_worker_4_switch()
    print(my_topo.topo)
    generate_json(my_topo.topo, '../data/topo/topo_7_workers.json')
    my_topo = topo_9_worker_4_switch()
    print(my_topo.topo)
    generate_json(my_topo.topo, '../data/topo/topo_9_workers.json')
    my_topo = topo_11_worker_4_switch()
    print(my_topo.topo)
    generate_json(my_topo.topo, '../data/topo/topo_11_workers.json')
    my_topo = topo_13_worker_4_switch()
    print(my_topo.topo)
    generate_json(my_topo.topo, '../data/topo/topo_13_workers.json')
    my_topo = topo_15_worker_4_switch()
    print(my_topo.topo)
    generate_json(my_topo.topo, '../data/topo/topo_15_workers.json')
