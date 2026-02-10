import sys
sys.path.append('..')
from mcf import min_cost_for_mcf
import networkx as nx
from math import log

def test_():

    """
    Test topology with three alternative paths between source and destination:
    0-1-5, 0-1-2-5, and 0-3-4-5. All links have unit cost but different capacities.
    A demand of 10 flow units is sent from s to t to test the min_cost algorithm.
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
    b = {'0': 10, '5': -10}


    #commodities
    srcs = ['0']
    tgts = ['5']
    ds   = [10]

    flow_ij, used_paths = min_cost_for_mcf(
                G, '0', '5', 10,
                c_label="capacity",
                l_label='l',
    )
            
    for P, f in used_paths:
        print(f"Path: {P}, Flow: {f}")
if __name__ == '__main__':
    test_()
