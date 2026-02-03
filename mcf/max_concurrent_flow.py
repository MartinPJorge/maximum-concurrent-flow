import networkx as nx
import logging
import sys
from collections import defaultdict

logger = logging.getLogger(__name__)



def min_cost_nosplit(G, srcs, tgts, ds, j, c_label,
        t_label=None, t_fn=None, max_t=None, search_paths=None):
    """
    :param: G is the graph
    :param srcs:
    :param tgts:
    :param ds: demand of each commodity
    :param j: commodity number
    :param c_label: capacity label for each edge
    :param t_label: transit time label for each edge
    :param t_fn: transit time function receives edge dict
    :param max_t: maximum transit time of considered paths
    :param search_paths: list of path lists for each src-tgt

    :note: t_label and max_t are parameters used to cap the
    maximum travel time of the path, they are not used by default.
    If max_t is present, it is assumed that either
    t_fn or t_label is present (just one of them).

    :note: if paths is specified, t_lavel, t_fn and max_t are ignored

    :returns: path, min_costj(l)
    """

    # Computes min_cost_j(l) considering unsplittable flows
    sj, tj, dj = srcs[j], tgts[j], ds[j]

    # Prune edges without enought capacity: c(j)<d(j)
    prunedG = nx.subgraph_view(G,
        filter_edge=lambda u,v: G[u][v][c_label] >= dj)

    # Get shortest path using l as cost function
    if search_paths:
        # Get the shortest path among the list of paths with cost l
        pathsj, min_l_cost, min_path = search_paths[j], float('inf'), None
        for path_i in pathsj:
            l_cost = sum([G[u][v]['l'] for u,v in zip(path_i[:-1],path_i[1:])])
            if l_cost < min_l_cost:
                min_path = path_i
                min_l_cost = l_cost
        path = min_path
    else:
        if not max_t:
            path = nx.shortest_path(prunedG, source=sj, target=tj,
                    weight='l')
        else:
            # Filter out shortest path if it has travel time
            # larger than max_t
            path = None
            for path_i in nx.shortest_simple_paths(prunedG, source=sj, target=tj):
                travel_time = sum([
                        G[u][v][t_label] if not t_fn else t_fn(G[u][v])\
                        for u,v in zip(path_i[:-1],path_i[1:])])
                logger.info('travel time', travel_time, 'max:', max_t, 'for path_i', path_i)
                if travel_time <= max_t:
                    path = path_i
                    break

    return path, sum([G[u][v]['l'] for u,v in zip(path[:-1],path[1:])])
    




def max_concurrent_flow_nosplit(G, srcs, tgts, ds, delta, eps, c_label,
        t_label=None, t_fn=None, max_t=None, search_paths=None,
        log_level=logging.WARNING):
    """
    :param c_label: capacity label for each edge
    :param t_label: transit time label for each edge
    :param t_fn: transit time function receives edge dict
    :param max_t: maximum transit time of considered paths
    :param search_paths: list of path lists for each src-tgt
    :param log_level: level for logging

    :note: t_label and max_t are parameters used to cap the
    maximum travel time of the path, they are not used by default.
    If max_t is present, it is assumed that t_label is as well.

    :Note: this replaces the mcf function by shortest paths
           however, returned paths may differ for same
           commodities across iterations

    :note: if paths is specified, t_lavel, t_fn and max_t are ignored
    """

    # Set logging level
    logging.basicConfig(level=log_level)

    # Set the initial values of l in graph G
    l = {(u,v): {'l': delta/G[u][v][c_label]} for u,v,d in G.edges(data=True)}
    nx.set_edge_attributes(G, l)

    # Keep track of the l_i,j updates
    ls = [l]

    # Keep track of the flow sent at each phase for all commodities
    f = {
        -1: {
            len(ds)-1: {(u,v): 0 for u,v in G.edges}
        }
    }
    for u,v in G.edges:
        f[-1][len(ds)-1][v,u] = 0

    # Keep track of the paths used to steer flow at every iteration
    paths = {}

    # Phases - phase i
    i = 0
    stop = False
    while not stop:
        logger.info('i=',i)
        paths[i] = {j: None for j in range(len(ds))}
        f[i] = {} if i!=0 else f[-1]

        # Iteration - per commodity j
        for j in range(len(ds)):
            logger.info('i-1=',i-1)
            logger.info('j-1=',j-1)
            f[i][j] = f[i][j-1] if j!=0 else f[i-1][len(ds)-1]

            logger.info(f'I set i={i} j={j}')
            logger.info(f' @start f[{i}][{j}]=', f[i][j])

            # Send d(j) units of commodity j along the paths
            # given by min_costj(l_i,j-1)
            paths[i][j], min_costj\
                = min_cost_nosplit(G, srcs, tgts, ds, j, c_label,
                        t_label, t_fn, max_t, search_paths)
            for u,v in zip(paths[i][j][:-1], paths[i][j][1:]):
                f[i][j][u,v] += ds[j]
                f[i][j][v,u] += ds[j]

            # Compute l_i,j(e)
            l_ij = {(u,v): {'l': d['l']*(1+eps*f[i][j][u,v]/d[c_label])}\
                    for u,v,d in G.edges(data=True)}
            ls.append(l_ij)

            # Update the graph edge attributes for l
            nx.set_edge_attributes(G, l_ij)


            logger.info(f' @end f[{i}][{j}]=', f[i][j])
            logger.info(f' @end l[{i}][{j}]=', ls[-1])

        
        # procedure stops at the first phase t for which D(t)>=1
        Dt = sum([d['l']*d[c_label] for u,v,d in G.edges(data=True)])
        stop = Dt >= 1
        i += 1

    return f, paths



def lambda_max_concurrent_flow_nosplit(G, ds, c_label, paths):
    """
    This function computes the lambda of the max_concurrent
    flow problem without split. It receives the result from
      max_concurrent_flow_nosplit

    :returns: lambda = min_j f(j)/d(j), {j: f(j)/d(j)}
    """

    G_ = G.copy() # TODO: copy the graph

    # Path selected for each commodity: the one for i=0
    j_paths = {j: str(paths[0][j]) for j in range(len(ds))}

    # Store the lambdas for each commodity j
    lambdas = {j: 0 for j in range(len(ds))}

    for i in paths.keys():
        for j in range(len(ds)):
            if str(paths[i][j]) != j_paths[j]:
                continue


            # Check if the sent flow fits
            fits = True
            for u,v in zip(paths[i][j][:-1], paths[i][j][1:]):
                fits = fits and (G_[u][v][c_label] >= ds[j])

            logger.info(f'@lambda i={i} j={j} fits={fits}')

            # In case the sent flow fits, update the graph
            if fits:
                lambdas[j] += 1
                for u,v in zip(paths[i][j][:-1], paths[i][j][1:]):
                    G_[u][v][c_label] -= ds[j]


    logger.info('lambdassss', lambdas)
            
    return min(lambdas.values()), lambdas

def min_cost2(G, s, t, demand, c_label="capacity", l_label="l"):
    """
    Routes the demand from s to t using successive shortest paths.
    At each step, the minimum-cost path is selected, the maximum feasible
    flow is sent through it, and residual capacities are updated.

    Returns:
    - flow: routed flow per edge {(u, v): f_uv}
    - used_paths: list of (path, routed_flow)
    - total_cost: total cost sum_e l_e * f_e
    """

    # Copy the graph to avoid modifying the original one    
    R = G.copy()
    sent_flow = 0
    flow = defaultdict(float)
    total_cost = 0.0

    used_paths = []   
    # print('Routing demand:', demand, 'from', s, 'to', t)

    # print('Graph', G.edges(data=True))
    i = 0
    while sent_flow < demand:
        i += 1
        print(f'  i={i}')

        remaining = demand - sent_flow

        # 1) Remove edges with insufficient residual capacity
        eligible_edges_graph = nx.subgraph_view(
            R,
            filter_edge=lambda u, v: R[u][v][c_label] > 0
        )
        print("Remaining:", remaining)
        print("Edges con capacidad mayor que >0:", eligible_edges_graph.number_of_edges())
        for u, v, d in R.edges(data=True):
            if d["capacity"] <= 0:
                print("Edge:", u, v, d["capacity"])
        try:
            path = nx.shortest_path(
                eligible_edges_graph,
                source=s,
                target=t,
                weight=l_label
            )
            # print('Found path:', path)
        except nx.NetworkXNoPath:
            raise ValueError("No hay más caminos para enviar la demanda")

        print('path:' , path)

        # 3) Compute the path capacity (bottleneck)
        edges = list(zip(path[:-1], path[1:]))
        path_cap = min(R[u][v][c_label] for u, v in edges)

        # 4) Determine the amount of flow to send
        f = min(path_cap, remaining)

        # Store the path and the routed flow
        used_paths.append((path, f))

        # 5) Route the flow and update residual capacities
        for u, v in edges:
            flow[(u, v)] += f
            total_cost += f * R[u][v][l_label]
            R[u][v][c_label] -= f
            cap_after = R[u][v][c_label]

            # print(f"Edge {u} -> {v}: capacity -> {cap_after}")
        sent_flow += f

    return dict(flow), used_paths, total_cost

def min_cost(G, s, t, demand, c_label="capacity", l_label="l"):

    R = G.copy()

    shortest_paths_generator = nx.all_shortest_paths(R, s, t, weight=l_label)

    visited_shortest_paths = []      
    generator_count = {'count': 0}   

    flow = defaultdict(float)
    used_paths = []
    total_cost = 0.0
    sent_flow = 0.0

    success, visited_shortest_paths, sent_flow, total_cost = min_cost_aux(
        R, s, t, demand,
        visited_shortest_paths,
        shortest_paths_generator,
        generator_count,
        sent_flow,
        flow,
        used_paths,
        total_cost,
        c_label,
        l_label,
    )

    return dict(flow), used_paths, total_cost


def min_cost_aux(R, s, t, demand,visited_shortest_paths,shortest_paths_generator, generator_count, sent_flow, flow, used_paths, total_cost, c_label="capacity", l_label="l"):
    i = generator_count["count"]

    while sent_flow < demand:

        if i < len(visited_shortest_paths):
            path = visited_shortest_paths[i]
        else:
            try:
                path = next(shortest_paths_generator)  
            except StopIteration:
                return False, visited_shortest_paths, sent_flow, total_cost

            visited_shortest_paths.append(path)

        edges = list(zip(path[:-1], path[1:]))

        cap_max = min(R[u][v].get(c_label, 0.0) for (u, v) in edges)

        if cap_max <= 0:
            i += 1
            generator_count["count"] = i
            continue

        send = min(cap_max, demand - sent_flow)

        for (u, v) in edges:
            R[u][v][c_label] -= send
            flow[(u, v)] += send

        path_cost = sum(R[u][v].get(l_label, 0.0) for (u, v) in edges)
        total_cost += send * path_cost

        used_paths.append((path, send))
        sent_flow += send

        i += 1
        generator_count["count"] = i

    return True, visited_shortest_paths, sent_flow, total_cost



        
def min_cost_alternative(G, srcs, tgts, ds, j):
    """
    Computes the minimum-cost routing for commodity j by distributing its demand
    among the available edge-disjoint paths, prioritizing shorter paths and
    respecting their capacities.

    Returns the used paths with their assigned flow and the total routing cost.
    """

    s, t = srcs[j], tgts[j]
    demand = ds[j]

    # 1) Compute all edge-disjoint paths between source and target
    paths = list(nx.edge_disjoint_paths(G, s, t))
    if not paths:
        raise ValueError("No hay edge-disjoint paths")

    # 2) Compute the length and capacity of each path
    path_info = []
    for P in paths:
        length = sum(G[u][v]["l"] for u, v in zip(P[:-1], P[1:]))
        capacity = min(G[u][v]["capacity"] for u, v in zip(P[:-1], P[1:]))
        path_info.append((P, length, capacity))

    # 3) Sort paths by increasing length
    path_info.sort(key=lambda x: x[1])

    # 4) Distribute the demand along the paths
    requirement = demand
    route = []
    total_cost = 0

    for P, length, cap in path_info:
        if requirement == 0:
            break
        flow = min(cap, requirement)  
        route.append((P, flow))
        total_cost +=  length*flow
        requirement -= flow

    if requirement > 0:
        raise ValueError("No hay capacidad suficiente para la demanda")

    return route, total_cost

def max_concurrent_flow_split(
    G, srcs, tgts, ds, delta, eps, c_label,
    log_level=logging.WARNING
):
    """
    Maximum Concurrent Flow (splittable version).

    Returns:
      - f[i][j][(u,v)] : flow of commodity j on edge (u,v) at phase i
      - paths[i][j]    : list of (path, flow) used for commodity j at phase i
    """

    logging.basicConfig(level=log_level)

    # Initial edge lengths: l_{1,0}(e) = delta / c(e)
    l0 = {(u, v): {'l': delta / d[c_label]} for u, v, d in G.edges(data=True)}
    nx.set_edge_attributes(G, l0)
    # Keep track of the flow sent at each phase for all commodities
    f = {
        -1: {
            len(ds)-1: {(u,v): 0 for u,v in G.edges}
        }
    }
    for u,v in G.edges:
        f[-1][len(ds)-1][v,u] = 0

    paths = {}

    i = 0
    stop = False

    while not stop:
        logger.info('i=',i)
        paths[i] = {j: None for j in range(len(ds))}
        f[i] = {} if i!=0 else f[-1]
    
        # Iteration - per commodity j
        for j in range(len(ds)):
            s, t = srcs[j], tgts[j]
            demand = ds[j]

            logger.info('i-1=',i-1)
            logger.info('j-1=',j-1)
            f[i][j] = f[i][j-1] if j!=0 else f[i-1][len(ds)-1]

            logger.info(f'I set i={i} j={j}')
            logger.info(f' @start f[{i}][{j}]=', f[i][j])

        # Iteration per commodity j
            # Solve min_cost_j(l_{i,j-1}) — SPLIT
            flow_ij, used_paths, _ = min_cost(
                G,
                s,
                t,
                demand,
                c_label=c_label,
                l_label='l'
            )
            # print('used paths[0]', used_paths[0])
            for path, flow in used_paths:
                for u, v in zip(path[:-1], path[1:]):
                    if (u,v) not in G.edges:
                        u, v = v, u # reverse edge
                    f[i][j][u,v] += flow

            paths[i][j] = used_paths
            # Update lengths l_{i,j}(e)
            new_l = {}
            for u, v, d in G.edges(data=True):
                #fij = flow_ij.get((u, v), 0.0)
                fij = f[i][j][u,v]
                new_l[(u, v)] = {
                    'l': d['l'] * (1 + eps * fij / d[c_label])
                }

            nx.set_edge_attributes(G, new_l)
        # Stopping condition: D(l) >= 1
        D = sum(d['l'] * d[c_label] for _, _, d in G.edges(data=True))
        stop = D >= 1
        i += 1

    return f, paths

def lambda_max_concurrent_flow_split(G, ds, c_label, paths):

    # Copy of the graph to simulate residual capacities
    G_ = G.copy()

    # Total flow effectively accepted per commodity
    flow_sent = {j: 0.0 for j in range(len(ds))}
    fitted_flow = {j: {} for j in range(len(ds))}

    # print("Initial capacities:")
    # for u, v, d in G_.edges(data=True):
    #     print(f"  {u}-{v}: {d[c_label]}")

    for i in paths.keys():
    #     print(f"\n Phase {i}")

        for j in range(len(ds)):
        #     print(f" Commodity {j} demand={ds[j]}")

            used_paths = paths[i][j]  # list of (path_nodes, flow)

            for path_nodes, flow in used_paths:

                edges = list(zip(path_nodes[:-1], path_nodes[1:]))

                min_cap = min(G_[u][v][c_label] for u, v in edges)

                fitted = min(flow, min_cap)

                # print(f"  Path {' -> '.join(map(str,path_nodes))} ,requested: {flow} , fitted: {fitted}")

                if fitted > 0:
                    # update residual capacities
                    for u, v in edges:
                        G_[u][v][c_label] -= fitted

                    # accumulate total sent for commodity j
                    flow_sent[j] += fitted

                # Keep track of the fitted flow along the edges
                edges_hash = hash(str(edges))
                if edges_hash not in fitted_flow[j]:
                    fitted_flow[j][edges_hash] = {
                        'edges': edges,
                        'flow': 0
                    }
                fitted_flow[j][edges_hash]['flow'] += fitted
          
        # for u, v, d in G_.edges(data=True):
        #     print(f"  {u}-{v}: {d[c_label]}")

    # Lambda value per commodity
    lambdas = {
        j: (flow_sent[j] / ds[j] if ds[j] > 0 else 0.0)
        for j in range(len(ds))
    }
    lambda_star = min(lambdas.values()) if lambdas else 0.0


    return lambda_star, lambdas, fitted_flow
