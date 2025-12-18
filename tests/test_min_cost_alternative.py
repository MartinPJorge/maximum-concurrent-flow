import sys
from mcf import min_cost_alternative
import networkx as nx
from math import log

def test_():

    """
    Test topology with three nodes (0-1-2) and two paths between source and destination.
    There is a direct path 0-2 with lower cost and an alternative path 0-1-2
    with higher cost but different capacity. A demand of 7 flow units
    is sent from s to t to test the min_cost_alternative algorithm.
    """

    G = nx.Graph()

    # Path P1: s -> o -> t (coste total = 2)
    G.add_edge('0', '1', l=1, capacity=10)
    G.add_edge('1', '2', l=1, capacity=4)

    # Path P2: s -> t (coste total = 1)
    G.add_edge('0', '2', l=1, capacity=5)
    srcs = ['0']
    tgts = ['2']
    ds   = [7]


    path, cost = min_cost_alternative(G, srcs, tgts, ds, j=0)

    print("Flow per edge:", path)
    print("Cost:", cost)


if __name__ == '__main__':
    test_()
