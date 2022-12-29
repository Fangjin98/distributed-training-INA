import json
import random

from collections import defaultdict

def get_link_array(paths: list , topo_dict: dict):
    link_index=defaultdict(dict)
    link_num=0
    band=[]
    
    for p in paths:
        weight_p=p.get_link_weight(topo_dict)
        for l in p.link_list:
            if l[1] not in link_index[l[0]].keys():
                link_index[l[0]][l[1]]=link_num
                link_num+=1
                band.append(weight_p[l[0]][l[1]])
    
    return link_index,band,link_num

class TopoGenerator(object):
    def __init__(self, topo_dict=dict()):
        self.topo_dict = dict(topo_dict)
        self.host_set=[]
        self.switch_set=[]
        
        for key in topo_dict.keys():
            for value in topo_dict[key]:
                try:
                    if key not in self.topo_dict[value]: # add reverse links
                        self.add_edge(value, key, topo_dict[key][value]) 
                except KeyError as e:
                    print(e)
                    self.topo_dict[value]=dict()
                    self.add_edge(value,key,topo_dict[key][value])
                    
        for key in self.topo_dict.keys():
            if key[0]=='v' and (key not in self.switch_set):
                self.switch_set.append(key)
            elif key[0]=='h'and (key not in self.host_set):
                self.host_set.append(key)
        
    def __str__(self) -> str:
        return str(self.topo_dict)

    def add_edge(self, src, dst, weight):
        self.topo_dict[src][dst]= weight

    def add_edges(self, edge_list):
        for e in edge_list:
            self.add_edge(e[0], e[1], e[2])

    def remove_edge(self, src, dst):
        if self.topo_dict[src]:
            for node in self.topo_dict[dst]:
                if node.keys()[0] is dst:
                    self.topo_dict[src].remove(node)
                    break
    
    def get_shortest_path(self,src,dst):
        return [Path(p) for p in self._get_feasible_path(src,dst,max_len=10)]

    
    def construct_path_set(self,src_set,dst_set,max_len=8):
        path=defaultdict(dict)
        for s in src_set:
            for d in dst_set:
                path[s][d]=[
                    Path(p) for p in self._get_feasible_path(s,d,max_len)]
        return path
    
    def _get_feasible_path(self, src, dst, max_len=None, path=[]):
        path=path+[src]

        if src == dst:
            return [path]

        if max_len is not None:
            if len(path) > max_len:
                return

        paths=[]

        for node in self.topo_dict[src].keys():
            if node not in path:
                results=self._get_feasible_path(node,dst,max_len,path)
                if results is not None:
                    for p in results:
                        paths.append(p)

        return paths
    
    def generate_json(self,json_file):
        json_str = json.dumps(self.topo_dict, indent=4)
        with open(json_file, 'w') as f:
            f.write(json_str)

    def generate_test_set(self, worker_num, switch_num,random_pick=False,seed=None):
        worker_num_per_rack=int(worker_num/switch_num)
        temp_host_set=list(self.host_set)
        temp_switch_set=list(self.switch_set)
        
        if random_pick:
            if not seed:
                random.seed(seed)
            random.shuffle(temp_host_set)
            random.shuffle(temp_switch_set)
        
        ps=temp_host_set[0]

        flatten_worker_set=[]
        switch_set=[]
        
        for i in range(switch_num):
            switch_set.append(temp_switch_set[i])
                 
        for i in range(1,worker_num+1):
                flatten_worker_set.append(temp_host_set[i])
        
        worker_set=[
            [flatten_worker_set[j*worker_num_per_rack+i] for i in range(worker_num_per_rack)]
             for j in range(switch_num)
        ]
        
        return [ps, worker_set, switch_set],[ps, flatten_worker_set, switch_set]

class Path(object):
    def __init__(self, node_list,link_list=None):
        self.node_list=node_list
        if link_list==None:
            self.link_list=[]
            for i in range(len(self.node_list)-1):
                node1=self.node_list[i]
                node2=self.node_list[i+1]
                self.link_list.append((node1,node2))
        else:
            self.link_list=link_list
            
    def __repr__(self):
        p_str=self.node_list[0]
        for node in self.node_list[1:]:
            p_str=p_str+'->'+ node
        return p_str
        
    def get_link_weight(self,topo_dict):
        link_weight=defaultdict(dict)
        for i in range(len(self.node_list)-1):
            node1=self.node_list[i]
            node2=self.node_list[i+1]
            link_weight[node1][node2]=topo_dict[node1][node2]
        return link_weight
