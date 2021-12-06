import json
import random
from collections import defaultdict
from collections import deque

import pulp as pl


def init_topo(topofile):
    link_set = []
    topo = json.load(open(topofile))
    node_list = list(topo.keys())
    network = defaultdict(dict)

    for node in node_list:
        for index, dst in enumerate(topo[node]):
            for dst_node, value in dst.items():
                link_set.append([node, dst_node])
                network[node][dst_node] = value
                network[dst_node][node] = value

    return network, link_set


def get_feasible_path(topo, src, dst, path=[]):
    path = path + [src]
    if src == dst:
        return [path]

    paths = []
    for node in topo[src].keys():
        if node not in path:
            ns = get_feasible_path(topo, node, dst, path)
            for n in ns:
                paths.append(n)

    return paths


def random_pick(elements, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(elements, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            return item


def covert_path(path):
    tmp = deque()
    coverted_path = []
    for node in path:
        tmp.append(node)
    for i in range(len(path)):
        coverted_path.append(tmp.pop())
    return coverted_path


def solve_lp(worker_num, switch_num,
             host_set, switch_set, path_set, link_set,
             switch_capacity, band, ps_num=1, t=96, mu=0.1):
    x = [[pl.LpVariable('x_n' + str(i) + '^s' + str(j), lowBound=0, upBound=1, cat=pl.LpContinuous)
          for j in range(switch_num + ps_num)]
         for i in range(worker_num)]
    y = [pl.LpVariable('y_s' + str(i), lowBound=0, upBound=1, cat=pl.LpContinuous)
         for i in range(switch_num)]
    V = [pl.LpVariable('V_s' + str(i), lowBound=0, cat=pl.LpContinuous)
         for i in range(switch_num)]

    path_set_index = [[[] for j in range(switch_num + ps_num)]
                      for i in range(worker_num + ps_num)]
    index_to_path = []

    path_count = 0
    for i in range(worker_num + ps_num):
        for j in range(switch_num):  # worker+ps->switch
            for k in range(len(path_set[host_set[i]][switch_set[j]])):
                path_set_index[i][j].append(path_count)
                index_to_path.append((host_set[i], switch_set[j], k))
                path_count += 1
        if i == worker_num + ps_num - ps_num:
            continue
        for j in range(ps_num):  # worker->ps
            for k in range(len(path_set[host_set[i]][host_set[worker_num + j]])):
                path_set_index[i][switch_num + j].append(path_count)
                index_to_path.append((host_set[i], host_set[worker_num + j], k))
                path_count += 1

    ep = [pl.LpVariable('epsilon_p' + str(i), lowBound=0, upBound=1, cat=pl.LpContinuous)
          for i in range(path_count)]

    I = [[0 for j in range(path_count)]
         for i in range(len(link_set))]

    for i in range(len(link_set)):
        for j in range(path_count):
            nodes = path_set[index_to_path[j][0]][index_to_path[j][1]][index_to_path[j][2]]
            n1, n2 = link_set[i]
            n1_in, n2_in = False, False
            for node in nodes:
                if n1 is node:
                    n1_in = True
                if n2 is node:
                    n2_in = True
            if n1_in and n2_in:
                I[i][j] = 1

    prob = pl.LpProblem("IAR", pl.LpMinimize)

    prob += pl.lpSum([x[i][switch_num] * t * mu for i in range(worker_num)])  # Objective
    # Constraints
    for i in range(switch_num):
        prob += V[i] == pl.lpSum([x[i][j] for i in range(worker_num) for j in range(switch_num)]) - 1

    for i in range(worker_num):
        for j in range(switch_num):
            prob += x[i][j] <= y[j]

    for i in range(worker_num):
        prob += pl.lpSum([x[i][j] for j in range(switch_num + ps_num)]) == 1

    for i in range(worker_num):
        for j in range(switch_num + ps_num):  # worker->switch+ps
            prob += pl.lpSum(ep[k] for k in path_set_index[i][j]) == x[i][j]

    for i in range(switch_num):
        for j in range(ps_num):  # switch->ps
            prob += pl.lpSum(ep[k] for k in path_set_index[worker_num + j][i]) == y[i]

    for i in range(switch_num):
        prob += V[i] * t * mu <= switch_capacity[i]

    for i in range(len(link_set)):
        prob += pl.lpSum([ep[m] * I[i][m] for j in range(worker_num)
                          for k in range(switch_num + ps_num)
                          for m in path_set_index[j][k]]) + \
                pl.lpSum([ep[m] * I[i][m] for j in range(switch_num)
                          for k in range(ps_num)
                          for m in path_set_index[worker_num + k][j]]) <= band[i] / t

    # status = prob.solve(pl.get_solver("CPLEX_PY"))
    status = prob.solve()
    print('objective =', pl.value(prob.objective))

    # f = open(
    #     "../data/lp1_x_user_" + str(USERAMOUNT) + "_request_" + str(request_num) + "_k_" + str(_K) + "_p_" + str(_P),
    #     "w")
    # f.write(str(pl.value(prob.objective)) + '\n')
    # for i in range(ECNAMOUNT):
    #     for j in range(VNFAMOUNT):
    #         f.write(str(pl.value(x[i][j])) + ' ')
    #     f.write('\n')
    # f.close()
    #
    # f = open(
    #     "../data/lp1_y_user_" + str(USERAMOUNT) + "_request_" + str(request_num) + "_k_" + str(_K) + "_p_" + str(_P),
    #     "w")
    # for i in range(ECNAMOUNT):
    #     for j in range(VNFAMOUNT):
    #         for k in range(USERAMOUNT):
    #             f.write(str(pl.value(y[i][j][k])) + ' ')
    #         f.write('\n')
    # f.close()

    x_res = [[pl.value(x[i][j]) for j in range(switch_num + ps_num)] for i in range(worker_num)]
    # y_res = [pl.value(y[i]) for i in range(switch_num)]
    # ep_res = [pl.value(ep[i]) for i in range(path_count)]

    return x_res


def RRIAR(worker_num, switch_num,
          host_set, switch_set, path_set, link_set,
          switch_capacity, band, ps_num=1, mu=0.1, t=96,
          file_name=None):
    x_res = solve_lp(worker_num, switch_num,
                     host_set, switch_set, path_set, link_set,
                     switch_capacity, band, ps_num, mu, t)
    s = [-1 for i in range(worker_num)]
    y = [0 for i in range(switch_num + ps_num)]
    path_list = []
    for i in range(worker_num):
        s[i] = random_pick([j for j in range(switch_num + ps_num)], x_res[i])
        y[s[i]] = 1
        # TODO: extend to multi paths
        if s[i] < switch_num:
            path_list.append(path_set[host_set[i]][switch_set[s[i]]][0])  # path of worker to switch
        else:
            path_list.append(path_set[host_set[i]][host_set[worker_num]][0])  # path of worker to ps
    for i in range(switch_num):
        if y[i] == 1:
            path_list.append(path_set[host_set[worker_num]][switch_set[i]][0])

    if file_name is not None:
        with open(file_name, 'w') as f:
            f.write('PS number = {}\n'
                    'Worker number = {}\n'
                    'Switch number = {}\n'
                    'Model size = {} Mb\n'
                    'Link bandwidth = {} Mbps\n'.format(
                str(ps_num), str(worker_num), str(switch_num), str(t), str(band[0])))
            f.write('\n')
            for i in range(worker_num):
                if s[i] < switch_num:
                    f.write(host_set[i] + ' aggregation-point ' + switch_set[s[i]])
                    f.write('\n')
                    f.write('Path: ' + str(path_set[host_set[i]][switch_set[s[i]]][0]))
                    f.write('\n')
                else:
                    f.write(host_set[i] + ' aggregation-point ' + host_set[worker_num])
                    f.write('\n')
                    f.write('Path: ' + str(path_set[host_set[i]][host_set[worker_num]][0]))
                    f.write('\n')
            for i in range(switch_num):
                if y[i] == 1:
                    f.write(switch_set[i] + ' aggregates')
                    f.write('\n')
                    f.write('Path: ' + str(covert_path(path_set[host_set[worker_num]][switch_set[i]][0])))
                    f.write('\n')
    return path_list


def schemes(worker_num):
    switch_set = ['s1', 's2', 's3', 's4','s5','s6']
    host_set = ['h' + str(i) for i in range(1, worker_num + 1)]
    ps_num = 1
    topo, link_set = init_topo('../data/topo/topo_{}_workers.json'.format(str(worker_num)))
    path_set = defaultdict(dict)
    for h in host_set:
        for s in switch_set:
            paths = get_feasible_path(topo, h, s)
            path_set[h][s] = paths
        for h1 in host_set:
            if h is h1:
                continue
            paths = get_feasible_path(topo, h, h1)
            path_set[h][h1] = paths
    for s in switch_set:
        for s1 in switch_set:
            if s is s1:
                continue
            paths = get_feasible_path(topo, s, s1)
            path_set[s][s1] = paths

    band = [1000 for i in range(len(link_set))]
    capacity = [400 for i in range(len(switch_set))]
    t = 100
    RRIAR(len(host_set) - ps_num, len(switch_set), host_set, switch_set, path_set, link_set, capacity,
          band, file_name='../data/path_{}_workers.txt'.format(str(worker_num)))


if __name__ == '__main__':
    # switch_set = ['s1', 's2', 's3', 's4']
    # host_set = ['h' + str(i) for i in range(1, 8)]
    # ps_num = 1
    # topo, link_set = init_topo('../data/topo/topo_7_workers.json')
    # path_set = defaultdict(dict)
    # for h in host_set:
    #     for s in switch_set:
    #         paths = get_feasible_path(topo, h, s)
    #         path_set[h][s] = paths
    #     for h1 in host_set:
    #         if h is h1:
    #             continue
    #         paths = get_feasible_path(topo, h, h1)
    #         path_set[h][h1] = paths
    # band = [500 for i in range(len(link_set))]
    # capacity = [500 for i in range(len(switch_set))]
    # t = 100
    # RRIAR(len(host_set) - ps_num, len(switch_set), host_set, switch_set, path_set, link_set, capacity,
    #       band, file_name='../data/path_7_workers.txt')

    for i in [7, 10, 13, 16, 19]:
        schemes(worker_num=i)
