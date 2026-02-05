import sys
sys.path.append('..')
from mcf import min_cost_v2
import networkx as nx
from math import log

def test_():

    G = nx.Graph()

    G.add_edge("gNB", "rg", l=1, capacity=600)
    G.add_edge("ru", "UPF", l=1, capacity=600)

    # --- Lado izquierdo ---
    G.add_edge("rg", "r5", l=1, capacity=200)
    G.add_edge("rg", "r6", l=2, capacity=200)

    G.add_edge("r5", "r4", l=2, capacity=200)
    G.add_edge("r5", "r6", l=1, capacity=200)

    # --- Acceso al core ---
    G.add_edge("r4", "r1", l=2, capacity=200)
    G.add_edge("r6", "r2", l=1, capacity=200)

    # --- Core ---
    G.add_edge("r2", "r1", l=1, capacity=200)
    G.add_edge("r1", "r3", l=2, capacity=200)
    # (opcional)
    G.add_edge("r2", "r3", l=2, capacity=200)

    # --- Lado derecho ---
    G.add_edge("r1", "r7", l=1, capacity=200)
    G.add_edge("r7", "r8", l=1, capacity=200)
    G.add_edge("r8", "r9", l=2, capacity=200)

    G.add_edge("r3", "r9", l=2, capacity=200)

    G.add_edge("r8", "ru", l=1, capacity=200)
    G.add_edge("r9", "ru", l=2, capacity=200)
       # commodity
    srcs = ["gNB"]
    tgts = ["UPF"]
    ds = [300]

    flow, paths,cost = min_cost_v2(
        G,
        s=srcs[0],
        t=tgts[0],
        demand=ds[0],
        c_label="capacity",
        l_label="l"
    )
    for P, f in paths:
        print(" -> ".join(P), "flow:",f)
    print("Flow per edge:", flow)
    print("Cost:", cost)
  
    print("Final flow:", f)
  


if __name__ == '__main__':
    test_()
