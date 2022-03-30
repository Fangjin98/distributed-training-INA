import numpy as np
import pulp as pl

from routing.algs.BasicAlg import BasicAlg
from routing.utils.TopoGenerator import TopoGenerator


class GRID(BasicAlg):
    def __init__(self, topo: TopoGenerator) -> None:
        super().__init__(topo)

    def run(self, test_set, comp, band, solver=None):
        ps = test_set[0]
        worker_set = test_set[1]
        switch_set = test_set[2]

        optimal_results = self._solve_lp(worker_set, switch_set, comp, band, solver)

        return self._random_rounding(optimal_results, ps, worker_set, switch_set)

    @staticmethod
    def _solve_lp(worker_set, switch_set, comp, band, solver_type):
        worker_num = len(worker_set)
        switch_num = len(switch_set)

        # Decision Variables
        x_ps = [pl.LpVariable('x_n' + str(i) + 'ps', lowBound=0, upBound=1)
                for i in range(worker_num)]

        x_s = [[pl.LpVariable('x_n' + str(i) + 's' + str(j), lowBound=0, upBound=1)
                for j in range(switch_num)]
               for i in range(worker_num)]

        r_ps = [pl.LpVariable('r_n' + str(i) + 'ps', lowBound=0, upBound=1)
                for i in range(worker_num)]

        r_s = [[pl.LpVariable('r_n' + str(i) + 's' + str(j), lowBound=0, upBound=1)
                for j in range(switch_num)]
               for i in range(worker_num)]

        y = [pl.LpVariable('y_s' + str(i), lowBound=0, upBound=1)
             for i in range(switch_num)]

        lam = pl.LpVariable('lambda', lowBound=0)

        prob = pl.LpProblem("GRIA", pl.LpMaximize)

        # Objective
        prob += lam

        # Constraints

        for i in range(worker_num):
            prob += pl.lpSum([x_ps[i]] + [x_s[i][j] for j in range(switch_num)]) == 1

        for i in range(worker_num):
            prob += r_ps[i] <= x_ps[i]

        for i in range(worker_num):
            for j in range(switch_num):
                prob += r_s[i][j] <= x_s[i][j]

        for i in range(worker_num):
            for j in range(switch_num):
                prob += r_s[i][j] <= y[j]

        for i in range(worker_num):
            prob += lam <= pl.lpSum([r_ps[i]] + [r_s[i][j] for j in range(switch_num)])

        for j in range(switch_num):
            prob += pl.lpSum([r_s[i][j] for i in range(worker_num)]) <= comp[j]

        prob += pl.lpSum([x_ps[i] for i in range(worker_num)]) + \
                pl.lpSum([y[j] for j in range(switch_num)]) <= band

        if solver_type is not None:
            try:
                prob.solve(pl.get_solver(solver_type))
                print("Using solver: " + solver_type)
            except Exception as e:
                print(e)
                prob.solve()
        else:
            prob.solve()

        print('objective =', pl.value(prob.objective))

        return np.asarray([x_ps[i].value() for i in range(worker_num)]), \
               np.asarray([[x_s[i][j].value() for j in range(switch_num)] for i in range(worker_num)]), \
               np.asarray([r_ps[i].value() for i in range(worker_num)]), \
               np.asarray([[r_s[i][j].value() for j in range(switch_num)] for i in range(worker_num)]), \
               np.asarray([y[i].value() for i in range(switch_num)])

    @staticmethod
    def _random_rounding(optimal_results, ps, worker_set, switch_set):
        x_ps = optimal_results[0]
        x_s = optimal_results[1]
        r_ps = optimal_results[2]
        r_s = optimal_results[3]
        y = optimal_results[4]

        ps_num = 1
        aggregation_node = dict()
        sending_rate = dict()
        prob = []
        switch = []

        for index, w in enumerate(worker_set):
            prob_x = np.concatenate(([x_ps[index]], x_s[index]))
            # if 0 < x_ps[index] < 1:  # HACK: avoid selecting ps
            #     s_res = np.argmax(x_s[index]) + ps_num  # directly select the switch with max prob
            # else:
            s_res = np.random.choice([i for i in range(ps_num + len(switch_set))], p=prob_x.ravel())

            if s_res < ps_num:  # aggregate on PS
                aggregation_node[w] = ps
                prob.append(prob_x[s_res])
            else:  # aggregate on switch
                aggregation_node[w] = switch_set[s_res - ps_num]
                prob.append(prob_x[s_res])
                switch.append(switch_set[s_res - ps_num])

        for index, w in enumerate(worker_set):
            if s_res < ps_num:  # aggregate on PS
                sending_rate[w] = r_ps[index] / x_ps[index]
            else:  # aggregate on switch
                sending_rate[w] = r_s[index][s_res - ps_num] / x_s[index][s_res - ps_num]

        for index, s in enumerate(switch_set):
            if s in switch:
                sending_rate[s] = max([r_s[i][index] for i in range(len(worker_set))]) / \
                                  np.sum([x_s[i][index] for i in range(len(worker_set))])

        return aggregation_node, sending_rate
