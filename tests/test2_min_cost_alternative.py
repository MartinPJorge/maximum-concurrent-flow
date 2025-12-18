import sys
from mcf import min_cost_alternative
import networkx as nx
from math import log

def test_():
    """
    Test topology with three alternative paths between source and destination:
    0-1-5, 0-1-2-5, and 0-3-4-5. All links have unit cost but different capacities.
    A demand of 7 flow units is sent from s to t to test the min_cost_alternative algorithm.
    """

    G = nx.Graph()

   # Path P1: 0 -> 1 -> 5
    G.add_edge('0', '1', l=1, capacity=10)
    G.add_edge('1', '5', l=1, capacity=4)

    # Path P2: 0 -> 1 -> 2->5 
    G.add_edge('0', '1', l=1, capacity=10)
    G.add_edge('1', '2', l=1, capacity=4)
    G.add_edge('2', '5', l=1, capacity=4)

    # Path P3: 0 ->3->4-> 5 
    G.add_edge('0', '3', l=1, capacity=5)
    G.add_edge('3', '4', l=1, capacity=5)
    G.add_edge('4', '5', l=1, capacity=5)

    #commodities
    srcs = ['0']
    tgts = ['5']
    ds   = [7]

    path, cost = min_cost_alternative(G, srcs, tgts, ds, j=0)

    print("Flow per edge:", path)
    print("Cost:", cost)


if __name__ == '__main__':
    test_()
