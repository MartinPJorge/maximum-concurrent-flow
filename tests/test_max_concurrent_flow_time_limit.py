import sys
sys.path.append('..')
from mcf import max_concurrent_flow
import networkx as nx
from math import log

def test_():

    G = nx.Graph()
    G.add_edge(0, 2, c=10, t=1)
    G.add_edge(1, 2, c=10, t=1)
    G.add_edge(2, 3, c=10, t=1)
    G.add_edge(2, 4, c=1000, t=1)
    G.add_edge(4, 3, c=1000, t=1)

    # commodities
    srcs = [0, 1]
    tgts = [3, 3]
    ds   = [3, 2]
    eps = 0.1
    m = len(G.edges)
    delta = (m / (1-eps))**(-1/eps)
    c_label = 'c'
    t_label = 't'
    max_t = 2

    f, paths = max_concurrent_flow.max_concurrent_flow_nosplit(
            G, srcs, tgts, ds, delta, eps, c_label,
            t_label=None,
            t_fn=lambda d: d[t_label],
            max_t=max_t)


    # Compute what is the lambda of the found solution

    lambda_, lambdas = max_concurrent_flow.lambda_max_concurrent_flow_nosplit(G, ds, c_label, paths)
    print('lambda=', lambda_)


    for j in range(len(ds)):
        print('paths of commodity', j)
        for i in range(len(paths.keys())):
            print('path', i, paths[i][j])




if __name__ == '__main__':
    test_()

