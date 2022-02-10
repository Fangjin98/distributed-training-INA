from collections import defaultdict
from collections import defaultdict
import json
import numpy as np
import sys
import os

sys.path.append("/home/sdn/fj/distributed_PS_ML")

from routing.utils.TopoGenerator import TopoGenerator
from routing.algs.RRIAR import RRIAR

if __name__=="__main__":
    np.random.seed(0)
    topo=TopoGenerator(json.load(open('/home/sdn/fj/distributed_PS_ML/routing/data/topo/fattree80.json')))
    myalg=RRIAR(topo)
    paths=myalg.run(ps_set=['h1'],worker_set=['h2','h3','h4','h5'],switch_set=['v1','v2','v3'])
    print(paths)