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


def topo_7_worker_6_switch():
    topo = Topo(defaultdict(list))
    topo.add_edges([('s1', 's2', 1), ('s1', 's3', 1), ('s1', 's4', 1)])
    topo.add_edges([('s5', 's2', 1), ('s5', 's3', 1), ('s5', 's4', 1)])
    topo.add_edges([('s6', 's2', 1), ('s6', 's3', 1), ('s6', 's4', 1)])
    topo.add_edges(
        [('s2', 'h1', 1), ('s2', 'h2', 1)])
    topo.add_edges(
        [('s3', 'h3', 1), ('s3', 'h4', 1)])
    topo.add_edges(
        [('s4', 'h5', 1), ('s4', 'h6', 1),('s4', 'h7', 1)])
    return topo


def topo_10_worker_6_switch():
    topo = Topo(defaultdict(list))
    topo.add_edges([('s1', 's2', 1), ('s1', 's3', 1), ('s1', 's4', 1)])
    topo.add_edges([('s5', 's2', 1), ('s5', 's3', 1), ('s5', 's4', 1)])
    topo.add_edges([('s6', 's2', 1), ('s6', 's3', 1), ('s6', 's4', 1)])
    topo.add_edges(
        [('s2', 'h1', 1), ('s2', 'h2', 1), ('s2', 'h3', 1)])
    topo.add_edges(
        [('s3', 'h4', 1), ('s3', 'h5', 1), ('s3', 'h6', 1)])
    topo.add_edges(
        [('s4', 'h7', 1), ('s4', 'h8', 1), ('s4', 'h9', 1), ('s4', 'h10', 1)])
    return topo


def topo_13_worker_6_switch():
    topo = Topo(defaultdict(list))
    topo.add_edges([('s1', 's2', 1), ('s1', 's3', 1), ('s1', 's4', 1)])
    topo.add_edges([('s5', 's2', 1), ('s5', 's3', 1), ('s5', 's4', 1)])
    topo.add_edges([('s6', 's2', 1), ('s6', 's3', 1), ('s6', 's4', 1)])
    topo.add_edges(
        [('s2', 'h1', 1), ('s2', 'h2', 1), ('s2', 'h3', 1), ('s2', 'h4', 1)])
    topo.add_edges(
        [('s3', 'h5', 1), ('s3', 'h6', 1), ('s3', 'h7', 1), ('s3', 'h8', 1)])
    topo.add_edges(
        [('s4', 'h9', 1), ('s4', 'h10', 1), ('s4', 'h11', 1), ('s4', 'h12', 1), ('s4', 'h13', 1)])
    return topo


def topo_16_worker_6_switch():
    topo = Topo(defaultdict(list))
    topo.add_edges([('s1', 's2', 1), ('s1', 's3', 1), ('s1', 's4', 1)])
    topo.add_edges([('s5', 's2', 1), ('s5', 's3', 1), ('s5', 's4', 1)])
    topo.add_edges([('s6', 's2', 1), ('s6', 's3', 1), ('s6', 's4', 1)])
    topo.add_edges(
        [('s2', 'h1', 1), ('s2', 'h2', 1), ('s2', 'h3', 1), ('s2', 'h4', 1),('s2', 'h5', 1)])
    topo.add_edges(
        [('s3', 'h6', 1), ('s3', 'h7', 1), ('s3', 'h8', 1),('s3', 'h9', 1), ('s3', 'h10', 1)])
    topo.add_edges(
        [('s4', 'h11', 1), ('s4', 'h12', 1), ('s4', 'h13', 1), ('s4', 'h14', 1), ('s4', 'h15', 1),('s4', 'h16', 1)])
    return topo


def topo_19_worker_6_switch():
    topo = Topo(defaultdict(list))
    topo.add_edges([('s1', 's2', 1), ('s1', 's3', 1), ('s1', 's4', 1)])
    topo.add_edges([('s5', 's2', 1), ('s5', 's3', 1), ('s5', 's4', 1)])
    topo.add_edges([('s6', 's2', 1), ('s6', 's3', 1), ('s6', 's4', 1)])
    topo.add_edges(
        [('s2', 'h1', 1), ('s2', 'h2', 1), ('s2', 'h3', 1), ('s2', 'h4', 1), ('s2', 'h5', 1), ('s2', 'h6', 1)])
    topo.add_edges(
        [('s3', 'h7', 1), ('s3', 'h8', 1), ('s3', 'h9', 1), ('s3', 'h10', 1), ('s3', 'h11', 1), ('s3', 'h12', 1)])
    topo.add_edges(
        [('s4', 'h13', 1), ('s4', 'h14', 1), ('s4', 'h15', 1), ('s4', 'h16', 1), ('s4', 'h17', 1), ('s4', 'h18', 1),('s4', 'h19', 1),])
    return topo



if __name__ == '__main__':
    my_topo = topo_7_worker_6_switch()
    print(my_topo.topo)
    generate_json(my_topo.topo, '../data/topo/topo_7_workers.json')
    my_topo = topo_10_worker_6_switch()
    print(my_topo.topo)
    generate_json(my_topo.topo, '../data/topo/topo_10_workers.json')
    my_topo = topo_13_worker_6_switch()
    print(my_topo.topo)
    generate_json(my_topo.topo, '../data/topo/topo_13_workers.json')
    my_topo = topo_16_worker_6_switch()
    print(my_topo.topo)
    generate_json(my_topo.topo, '../data/topo/topo_16_workers.json')
    my_topo = topo_19_worker_6_switch()
    print(my_topo.topo)
    generate_json(my_topo.topo, '../data/topo/topo_19_workers.json')
