import networkx as nx



def min_cost_nosplit(G, srcs, tgts, ds, j, c_label):
    """
    :param: G is the graph
    :param srcs:
    :param tgts:
    :param ds: demand of each commodity
    :param j: commodity number
    :param c_label: capacity label for each edge

    :returns: path, min_costj(l)
    """
    # TODO: cap the paths longer than T*

    # Computes min_cost_j(l) considering unsplittable flows
    sj, tj, dj = srcs[j], tgts[j], ds[j]

    # Prune edges without enought capacity: c(j)<d(j)
    prunedG = nx.subgraph_view(G,
        filter_edge=lambda u,v: G[u][v][c_label] >= dj)

    # Get shortest path using l as cost function
    path = nx.shortest_path(prunedG, source=sj, target=tj,
            weight='l')

    return path, sum([G[u][v]['l'] for u,v in zip(path[:-1],path[1:])])
    




def max_concurrent_flow_nosplit(G, srcs, tgts, ds, delta, eps, c_label):
    """
    :param c_label: capacity label for each edge

    :Note: this replaces the mcf function by shortest paths
           however, returned paths may differ for same
           commodities across iterations
    """


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
        print('i=',i)
        paths[i] = {j: None for j in range(len(ds))}
        f[i] = {} if i!=0 else f[-1]

        # Iteration - per commodity j
        for j in range(len(ds)):
            print('i-1=',i-1)
            print('j-1=',j-1)
            f[i][j] = f[i][j-1] if j!=0 else f[i-1][len(ds)-1]

            print(f'I set i={i} j={j}')
            print(f' @start f[{i}][{j}]=', f[i][j])

            # Send d(j) units of commodity j along the paths
            # given by min_costj(l_i,j-1)
            paths[i][j], min_costj\
                = min_cost_nosplit(G, srcs, tgts, ds, j, c_label)
            for u,v in zip(paths[i][j][:-1], paths[i][j][1:]):
                f[i][j][u,v] += ds[j]
                f[i][j][v,u] += ds[j]

            # Compute l_i,j(e)
            l_ij = {(u,v): {'l': d['l']*(1+eps*f[i][j][u,v]/d[c_label])}\
                    for u,v,d in G.edges(data=True)}
            ls.append(l_ij)

            # Update the graph edge attributes for l
            nx.set_edge_attributes(G, l_ij)


            print(f' @end f[{i}][{j}]=', f[i][j])
            print(f' @end l[{i}][{j}]=', ls[-1])

        
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

    :returns: lambda = min_j f(j)/d(j)
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

            print(f'@lambda i={i} j={j} fits={fits}')

            # In case the sent flow fits, update the graph
            if fits:
                lambdas[j] += 1
                for u,v in zip(paths[i][j][:-1], paths[i][j][1:]):
                    G_[u][v][c_label] -= ds[j]


    print('lambdassss', lambdas)
            
    return min(lambdas.values())

