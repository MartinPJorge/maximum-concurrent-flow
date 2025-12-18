import sys
from mcf import  max_concurrent_flow_split, lambda_max_concurrent_flow_split, min_cost
import networkx as nx
from math import log

def test_():

    """
    Test topology with two alternative paths between source and destination:
    0 -> 2 and 0 -> 1 -> 2. All links have unit cost but different capacities.
    Three commodities with demands [2, 1, 2] are routed to compare
    independent min-cost routing with the Maximum Concurrent Flow (splittable) algorithm.
    """
    G = nx.Graph()

   # Path P1: 0 -> 2
    G.add_edge('0', '2', l=1, capacity=4)

    # Path P2: 0 -> 1 ->2
    G.add_edge('0', '1', l=1, capacity=1)
    G.add_edge('1', '2', l=1, capacity=1)

    #commodities
    srcs = ['0','0','0']
    tgts = ['2','2','2']
    ds   = [2,1,2]
    eps=0.1
    m = len(G.edges)
    delta = (m / (1-eps))**(-1/eps)

    for j in range(len(ds)):
        flow, paths,cost = min_cost(
            G,
            s=srcs[j],
            t=tgts[j],
            demand=ds[j],
            c_label="capacity",
            l_label="l"
        )
        for P, f in paths:
            print(" -> ".join(P), "flow:",f)
        print("Flow per edge:", flow)
        print("Cost:", cost)
    
        print("Final flow:", f)

    f, paths = max_concurrent_flow_split(G,
        srcs,
        tgts,
        ds,
        delta,
        eps,
        c_label="capacity")

    for i in paths:
        for j in paths[i]:
            print(f"Phase {i}, commodity {j}")
            for P, flow in paths[i][j]:
                print("  ", " -> ".join(P), "flow:", flow)


    lambda_star, lambdas = lambda_max_concurrent_flow_split(
    G=G,
    ds=ds,
    c_label="capacity",
    paths=paths
)

    print("λ global:", lambda_star)
    print("λ por commodity:", lambdas)


if __name__ == '__main__':
    test_()

    