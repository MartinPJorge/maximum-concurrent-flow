import sys
sys.path.append('..')

from mcf import min_cost_for_mcf
import networkx as nx
from math import log

def test_():

    """
    Test topology with three nodes (0-1-2) and two paths between source and destination.
    There is a direct path 0-2 with lower cost and an alternative path 0-1-2
    with higher cost but different capacity. A demand of 7 flow units
    is sent from s to t to test the min_cost algorithm.
    """
    G = nx.Graph()

    # Path P1: 0 -> 1 -> 2 
    G.add_edge('0', '1', l=1, capacity=10)
    G.add_edge('1', '2', l=1, capacity=4)

    # Path P2: 0 -> 2 
    G.add_edge('0', '2', l=1, capacity=5)
    srcs = ['0']
    tgts = ['2']
    ds   = [7]

    flow_ij, used_paths = min_cost_for_mcf(
                G, '0', '2', 7,
                c_label="capacity",
                l_label='l',
    )
            
    for P, f in used_paths:
        print(f"Path: {P}, Flow: {f}")


if __name__ == '__main__':
    test_()
