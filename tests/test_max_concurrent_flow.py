import sys
sys.path.append('../src')
from mcf import max_concurrent_flow
import networkx as nx
from math import log

def test_():

    G = nx.Graph()
    G.add_edge(0, 1, c=10)
    G.add_edge(0, 2, c=10)
    G.add_edge(1, 2, c=10)
    G.add_edge(0, 3, c=10)
    G.add_edge(3, 2, c=10)

    # commodities
    srcs = [0, 3, 3]
    tgts = [2, 2, 2]
    ds   = [3, 2, 3]
    eps = 0.1
    m = len(G.edges)
    delta = (m / (1-eps))**(-1/eps)
    c_label = 'c'

    f, paths = max_concurrent_flow.max_concurrent_flow_nosplit(
            G, srcs, tgts, ds, delta, eps, c_label)


    # Compute what is the lambda of the found solution

    lambda_ = max_concurrent_flow.lambda_max_concurrent_flow_nosplit(G, ds, c_label, paths)
    print('lambda=', lambda_)


    for j in range(len(ds)):
        print('paths of commodity', j)
        for i in range(len(paths.keys())):
            print('path', i, paths[i][j])



    print('We have an (1-eps)^-3 approx', (1-eps)**(-3))

    print('I now scale the flow by log_(1+eps)(1/delta)',
            log(1/delta, 1+eps))
    i_last = max(list(f.keys()))
    print('i_last', i_last)
    f_ = {e: flow*log(1/delta, 1+eps)\
            for e,flow in f[i_last][2].items()}
    print('f_', f_)


if __name__ == '__main__':
    test_()

