import sys
from mcf import min_cost
import networkx as nx
from math import log

def test_():

    """
    Test topology with three nodes (s, o, t) and two paths between source and destination.
    There is a direct path s–t with lower cost and an alternative path s–o–t
    with higher cost but different capacity. A demand of 7 flow units
    is sent from s to t to test the min_cost algorithm.
    """

    G = nx.Graph()

    # Path P1: s -> o -> t 
    G.add_edge('s', 'o', l=1, capacity=10)
    G.add_edge('o', 't', l=1, capacity=4)

    # Path P2: s -> t (coste total = 1)
    G.add_edge('s', 't', l=1, capacity=5)
    srcs = ['s']
    tgts = ['t']
    ds   = [7]

    path, cost = min_cost(G, srcs, tgts, ds, j=0)

    print("Path elegido:", path)
    print("Coste:", cost)


if __name__ == '__main__':
    test_()
