import sys
from mcf import min_cost
import networkx as nx
from math import log

def test_():
    """
    Test topology with three alternative paths between source and destination:
    s–o–t, s–o–a–t, and s–b–c–t. All links have unit cost but different capacities.
    A demand of 7 flow units is sent from s to t to test the min_cost algorithm.
    """

    G = nx.Graph()

    # Path P1: s -> o -> t 
    G.add_edge('s', 'o', l=1, capacity=10)
    G.add_edge('o', 't', l=1, capacity=4)

    # Path P2: s -> o -> a->t 
    G.add_edge('s', 'o', l=1, capacity=10)
    G.add_edge('o', 'a', l=1, capacity=4)
    G.add_edge('a', 't', l=1, capacity=4)

    # Path P3: s ->b->c-> t 
    G.add_edge('s', 'b', l=1, capacity=5)
    G.add_edge('b', 'c', l=1, capacity=5)
    G.add_edge('c', 't', l=1, capacity=5)

    #commodities
    srcs = ['s']
    tgts = ['t']
    ds   = [7]

    path, cost = min_cost(G, srcs, tgts, ds, j=0)

    print("Path elegido:", path)
    print("Coste:", cost)


if __name__ == '__main__':
    test_()
