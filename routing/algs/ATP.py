from collections import defaultdict
from email.policy import default
import random
from typing import List
from routing.algs.BasicAlg import BasicAlg
from routing.utils.TopoGenerator import TopoGenerator


class ATP(BasicAlg):
    def __init__(self, topo: TopoGenerator, test_set, **kwargs) -> None:
        super().__init__(topo)
        
        self.ps=test_set[0]
        self.worker_set=test_set[1]
        self.switch_set=test_set[2]
        
        self.flatten_worker_set=[]
        for ww in self.worker_set:
            for w in ww:
                self.flatten_worker_set.append(w)
        
        self.comp=kwargs["comp"]
        self.t=kwargs["t"]
    
    def run(self):
        tor_switch_set=self._get_tor_switch()
        candidate_paths=self._get_candidate_path()
        capacity_dict={
            switch:val for switch,val in zip(self.switch_set,self.comp) 
        }
        
        routing_paths=[]
        aggregation_node=dict()
        switch_aggregatioin={k:False for k in self.switch_set}
            
        for w in self.flatten_worker_set:
            if capacity_dict[tor_switch_set[w]] >= self.t: # aggregate on tier 0
                
                capacity_dict[tor_switch_set[w]]-=self.t
                switch_aggregatioin[tor_switch_set[w]]=True

                aggregation_node[w]=tor_switch_set[w]
                
                routing_paths.append(
                    self._choose_path(candidate_paths[w][tor_switch_set[w]])
                ) 
                
            elif capacity_dict[tor_switch_set[self.ps]] >= self.t: # aggregate on tier 1
                capacity_dict[tor_switch_set[self.ps]]-=self.t
                switch_aggregatioin[tor_switch_set[self.ps]]=True
                
                aggregation_node[w]=tor_switch_set[self.ps]
                
                routing_paths.append(
                    self._choose_path(candidate_paths[w][tor_switch_set[self.ps]])
                ) 
                
            else:  # directly aggregate on ps
                aggregation_node[w]=self.ps
                
                try:
                    routing_paths.append(
                        self._choose_path(candidate_paths[w][self.ps])
                    ) 
                except Exception as e:
                    print(e)
        
        for s in self.switch_set:
            if switch_aggregatioin[s]:
                routing_paths.append(
                    self._choose_path(candidate_paths[s][self.ps])
                )
        
        return aggregation_node, routing_paths
            
    def _get_candidate_path(self):
        path_worker_ps = self.topo.construct_path_set(self.flatten_worker_set, [self.ps])
        path_worker_switch = self.topo.construct_path_set(self.flatten_worker_set,self.switch_set)
        path_switch_ps = self.topo.construct_path_set(self.switch_set, [self.ps])
        
        paths=defaultdict(dict)
        for w in self.flatten_worker_set:
            paths[w].update(path_worker_ps[w])
            paths[w].update(path_worker_switch[w])
        for s in self.switch_set:
            paths[s].update(path_switch_ps[s])
        
        return paths
    
    @staticmethod
    def _choose_path(path_set):
        # simply get the first paths of the path set.
        return random.choice(path_set)
        
    def _get_tor_switch(self):
        tor_switch=defaultdict(dict)
        
        for index,ww in enumerate(self.worker_set):
            for w in ww:
                tor_switch[w]=self.switch_set[index]
        
        tor_switch[self.ps]=self.switch_set[0]
        
        return tor_switch