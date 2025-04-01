import networkx as nx
import logging
import sys
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

