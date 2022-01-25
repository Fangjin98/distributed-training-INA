from collections import defaultdict
from collections import deque
import json

import pulp as pl
import numpy as np

from routing.utils.TopoGenerator import TopoGenerator

def get_link_array(paths):
    link_index=defaultdict(dict)
    link_num=0
    band=[]
    
    for p in paths:
        link_p=p.get_link()
        weight_p=p.get_link_weight()
        for l in link_p:
            if link_index[l[0]].keys(l[1]) is None:
                link_index[l[0]][l[1]]=link_num
                link_num+=1
                band.append(weight_p[l[0]][l[1]])
    
    return link_index,band,link_num

class RRIAR:
    def __init__(self,topo, ps_num=1) -> None:
        self.topo=topo
        self.ps_num=ps_num
        self.path_num=0
        self.worker_path_num=0
        self.switch_path_num=0

    def run(self,ps_set, worker_set, switch_set, t=90,mu=1):
        path_index, constant_I,band=self._get_constants(ps_set, worker_set, switch_set)
        comp=[1 for i in range(len(switch_set))]
        optimal_results=self._solve_lp(ps_set, worker_set,switch_set,path_index, constant_I,band,comp,t,mu)
        return self._random_rounding(optimal_results)

    def _get_constants(self,ps_set, worker_set,switch_set):
        path_worker_ps = self.topo.construct_path_set(worker_set, ps_set)
        path_worker_switch = self.topo.construct_path_set(worker_set,switch_set)
        path_switch_ps = self.topo.construct_path_set(switch_set, ps_set)
                
        path_index=defaultdict(dict)
        paths=[]
        path_num=0
        
        for w in worker_set:
            for ps in ps_set:
                paths+=path_worker_ps[w][ps]
                path_index[w][ps]=(path_num,path_num+len(path_worker_ps[w][ps]))
                path_num+=len(path_worker_ps[w][ps])
            
        for w in worker_set:
            for s in switch_set:
                paths+=path_worker_switch[w][s]
                path_index[w][s]=(path_num,path_num+len(path_worker_switch[w][s]))
                path_num+=len(path_worker_switch[w][s])

        self.worker_path_num=path_num
        
        for s in switch_set:
            for ps in ps_set:
                paths+=path_switch_ps[s][ps]
                path_index[s][ps]=(path_num,path_num+len(path_switch_ps[s][ps]))
                path_num+=len(path_switch_ps[s][ps])
        
        self.switch_path_num=path_num-self.worker_path_num
        self.path_num=path_num
        
        link_index,band,link_num=get_link_array(paths)
        
        constant_I=np.zeros((path_num,link_num))
        for p,index in enumerate(paths):
            link_p=p.get_link()
            for l in link_p:
                constant_I[index][link_index[l[0]][l[1]]]=1
        
        return path_index, constant_I, band
                    
    def _solve_lp(self, ps_set, worker_set, switch_set, path_set, constant_I, band, comp, t, mu):
        ps_num=len(ps_set)
        worker_num=len(worker_set)
        switch_num=len(switch_set)
        path_num=len(path_set)
        link_num=len(band)

        # Decision Variables
        x_ps = [[pl.LpVariable('x_n' + str(i) + '^ps' + str(j), lowBound=0, upBound=1, cat=pl.LpContinuous)
            for j in range(ps_num)]
            for i in range(worker_num)]
        
        x_s = [[pl.LpVariable('x_n' + str(i) + '^s' + str(j), lowBound=0, upBound=1, cat=pl.LpContinuous)
            for j in range(switch_num)]
            for i in range(worker_num)]
        
        y = [pl.LpVariable('y_s' + str(i), lowBound=0, upBound=1, cat=pl.LpContinuous)
            for i in range(switch_num)]

        ep = [pl.LpVariable('epsilon_p' + str(i), lowBound=0, upBound=1, cat=pl.LpContinuous)
            for i in range(path_num)]

        vv = [pl.LpVariable('V_s' + str(i), lowBound=0, cat=pl.LpInteger)
            for i in range(switch_num)]

        prob = pl.LpProblem("IAR", pl.LpMinimize)

        # Objective
        prob += pl.lpSum([x_ps[i][j] * t * mu for i in range(worker_num) for j in range(ps_num)])  
        
        # Constraints
        for i in range(switch_num):
            prob += vv[i] == pl.lpSum([x_s[i][j] for i in range(worker_num) for j in range(switch_num)]) - 1

        for i in range(worker_num):
            for j in range(switch_num):
                prob += x_s[i][j] <= y[j]

        for i in range(worker_num):
            prob += pl.lpSum([x_ps[i][j] for j in range(ps_num)]+[x_s[i][j] for j in range(switch_num + ps_num)]) == 1

        for i in range(worker_num):
            for j in range(ps_num): 
                prob += pl.lpSum(ep[k] for k in range(path_set[worker_set[i]][ps_set[j]][0],path_set[worker_set[i]][ps_set[j]][1])) == x_ps[i][j]
        
        for i in range(worker_num):
            for j in range(switch_num): 
                prob += pl.lpSum(ep[k] for k in range(path_set[worker_set[i]][switch_set[j]][0],path_set[worker_set[i]][switch_set[j]][1])) == x_s[i][j]

        for i in range(switch_num):
            for j in range(ps_num): 
                prob += pl.lpSum(ep[k] for k in range(path_set[switch_set[i]][ps_set[j]][0],switch_set[worker_set[i]][ps_set[j]][1])) == y[i]

        for i in range(switch_num):
            prob += vv[i] * t * mu <= comp[i]

        for i in range(link_num):
            prob += (pl.lpSum([ep[m] * constant_I[m][i] for m in range(self.worker_path_num)]) + pl.lpSum([ep[m] * constant_I[m][i] for m in range(self.worker_path_num,self.switch_path_num)])) * t <= band[i]

        # status = prob.solve(pl.get_solver("CPLEX_PY"))
        status = prob.solve()
        print('objective =', pl.value(prob.objective))

        x_ps_res = [[pl.value(x_ps[i][j]) for j in range( ps_num)]for i in range(worker_num) ] 
        x_s_res=[[pl.value(x_s[i][j]) for j in range(switch_num)] for i in range(worker_num)]
        y_res = [pl.value(y[i]) for i in range(switch_num)]
        ep_res = [pl.value(ep[i]) for i in range(path_num)]

        return x_ps_res,x_s_res, y_res, ep_res

    def _random_rounding(optimal_results):
        pass

if __name__=="__main__":
    topo=TopoGenerator(json.load(open('/home/sdn/fj/distributed_PS_ML/routing/data/topo/fattree80.json')))
    myalg=RRIAR(topo)
    RRIAR.run(['h1'],['h2','h3','h4','h5'],['s1','s2','s3'])
    
