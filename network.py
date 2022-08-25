import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy
import sys
from collections import Counter
import json
from matplotlib.animation import PillowWriter,FuncAnimation


# Number of new nodes distribution per step (with selected distribution)
def generate_number_of_new_nodes(dist):
    if dist == "poisson":
        return np.random.poisson(lam=2.5)
    elif dist == "binomial":
        return np.random.binomial(n=10,p=0.25)
    elif dist == "exponential":
        return np.random.exponential(scale=7)
    elif dist == "pareto":
        return scipy.stats.pareto.rvs(1,scale=1)
    elif dist == "degenerate":
        return 1
    elif dist == "levy":
        return scipy.stats.levy.rvs(loc=0,scale=1)
    elif dist == "cauchy":
        return scipy.stats.cauchy.rvs(loc=2.5,scale=1)
    elif dist == "normal":
        return np.random.normal(loc=2.5,scale=4)
    else:
        raise Exception("Unknown distribution")

# Number of new edges distribution per new node (with selected distribution)
def generate_number_of_new_edges(dist):
    if dist == "exponential":
        return np.random.exponential()
    elif dist == "uniform":
        return np.random.uniform(low=1, high=7)
    elif dist == "constant":
        return 3 # Can by any constant value
    elif dist == "normal":
        return np.random.normal(loc=2.5,scale=4)
    else:
        raise Exception("Unknown distribution")

# Lifetime of a node (number of step for survival, with normal distribution)
def generate_lifetime_of_node():
    return np.random.normal(50,25)
   
# Search time for a node (with poisson distribution)
def generate_search_time_of_node():
    return np.random.poisson(lam=2.5)

# Animate function for drawing GIF from snapshots
def animate(i,all_G):
    plt.clf()
    ax = plt.gca()
    ax.set_title('Cycle #'+str(i)+' out of '+str(len(all_G)))
    pos = nx.circular_layout(all_G[i])
    nx.draw(all_G[i], pos=pos, with_labels=True)
    print("Generating GIF... (",i,"out of",len(all_G),")")
    
# Save snapshots of a graph as a GIF
def save_as_gif(all_G):
    fig = plt.figure(figsize=(8,8))
    ani = FuncAnimation(fig, animate, frames=len(all_G),fargs=(all_G,))
    ani.save('output.gif', writer=PillowWriter(fps=5))
    print("GIF saved in output.gif!")

#disply graph
def display_graph(G):
    plt.figure(figsize=(8,8))
    pos = nx.circular_layout(G)
    nx.draw(G, pos=pos, with_labels=True)
    plt.show()



# Display graph on screen
def display_graph(G):
    plt.figure(figsize=(8,8))
    pos = nx.circular_layout(G)
    nx.draw(G, pos=pos, with_labels=True)
    plt.show()

# Plot graph degree histogram with CI
def plot_degree_dist_with_ci(all_G):
    plt.clf()
    all_hist = []
    for G in all_G:
        degrees = [d for n,d in G.degree()]
        hist = [(x,degrees.count(x)) for x in set(degrees)]
        all_hist.append(hist)
    total_hist = {}
    for hist in all_hist:
        for item in hist:
            if item[0] not in total_hist:
                total_hist[item[0]] = [item[1],]
            else:
                total_hist[item[0]].append(item[1])
    total_hist = sorted(list(total_hist.items()),key=lambda x:x[0])
    x = [item[0] for item in total_hist]
    y_mean = np.array([np.mean(item[1]) for item in total_hist])
    y_err = np.array([1.96*(np.std(item[1])/np.sqrt(len(item[1]))) for item in total_hist])
    plt.plot(x,y_mean)
    plt.fill_between(x,y_mean-y_err,y_mean+y_err,alpha=0.2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title("Degree Distribution")
    plt.show()

# Plot isolation probability regarding to cycles with CI
def plot_isolation_probability_with_ci(isolation_probability,cycle):
    plt.clf()
    x = list(range(0, cycle))
    y_mean = isolation_probability
    y_err = 1.96*(np.std(isolation_probability,axis=0)/np.sqrt(len(isolation_probability)))
    plt.plot(x,y_mean)
    plt.fill_between(x,y_mean-y_err,y_mean+y_err,alpha=0.2)
    plt.xlabel('Cycle')
    plt.ylabel('Isolation Probability')
    plt.title("Isolation Probability")
    plt.show()

# Modified Barabasi-Albert Model based on:
# 1) no zero degree node 
# 2) specifying lifetime for nodes
# 3) specifying attacks to the highest degree node in specific cycles
# 
# Parameters:
# m0: initial empty graph with m0 nodes (integer)
# nmax: maximum number of nodes to be added in every cycle (integer)
# cycles: number of cycles (integer)
# attacks: specificied cycles in which the highest degree node is going to be attacked (list of integers)
# generate_snapshot: output every cycle snapshot or just final output (boolean)
# new_nodes_distribution: distribution of new nodes (string)
# new_edges_distribution: distribution of new edges (string)
def modified_barabasi_albert(m0,nmax,cycles,attacks,attacks_type,connection_type,generate_snapshot,new_nodes_distribution,new_edges_distribution,active):
    G = nx.complete_graph(m0)
    nodes_endlife = {}
    nodes_to_be_connected_by_time = {}
    i = m0
    snapshots = []
    for cycle in range(cycles):
        number_of_nodes = G.number_of_nodes()
        number_of_edges = G.number_of_edges()
        degrees = G.degree()
        new_nodes = int(generate_number_of_new_nodes(new_nodes_distribution))
        while new_nodes > nmax:
            new_nodes = int(generate_number_of_new_nodes(new_nodes_distribution))
        nodes_to_be_added = []
        edges_to_be_added = []
        for node in range(new_nodes):
            new_edges = int(generate_number_of_new_edges(new_edges_distribution))
            while new_edges > number_of_nodes or new_edges <= 0:
                new_edges = int(generate_number_of_new_edges(new_edges_distribution))
            nodeid = i
            i += 1
            nodes_to_be_added.append(nodeid)
            lifetime = int(generate_lifetime_of_node())
            while lifetime <= 0:
                lifetime = int(generate_lifetime_of_node())
            endlife = cycle + lifetime
            if endlife not in nodes_endlife:
                nodes_endlife[endlife] = [nodeid,]
            else:
                nodes_endlife[endlife].append(nodeid)
            chosen_nodes = np.random.choice(a=[n for n,d in degrees],size=new_edges,replace=False,p=np.array([d+1 for n,d in degrees])/np.sum([d+1 for n,d in degrees]))
            edges_to_be_added += [(nodeid,n) for n in chosen_nodes]
        G.add_nodes_from(nodes_to_be_added)
        G.add_edges_from(edges_to_be_added)
        # Deleting dead nodes and replacing its edges
        if active==1:
            all_neighbor_nodes = []
            if cycle in nodes_endlife:
                for node in nodes_endlife[cycle]:
                    if node in G.nodes:
                        neighbor_nodes = G.neighbors(node)
                        for neighbor_node in neighbor_nodes:
                            all_neighbor_nodes.append(neighbor_node)

        if cycle in nodes_endlife:
            G.remove_nodes_from(nodes_endlife[cycle])

        if active==1:
            most_degree_node = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][0]
            for node in all_neighbor_nodes:
                search_time = int(generate_search_time_of_node())
                while search_time <= 0:
                    search_time = int(generate_search_time_of_node())
                edge_add_time = cycle + search_time
                if edge_add_time not in nodes_to_be_connected_by_time:
                    nodes_to_be_connected_by_time[edge_add_time] = [node,]
                else:
                    nodes_to_be_connected_by_time[edge_add_time].append(node)

        # Perform attack (delete node with highest degree)
        all_neighbor_nodes = []
        if cycle in attacks:
            if attacks_type == 'h':
                all_neighbor_nodes += G.neighbors(most_degree_node)
                G.remove_node(most_degree_node)
            elif attacks_type == 'r':
                random_node = np.random.choice(G.nodes())
                all_neighbor_nodes += G.neighbors(random_node)
                G.remove_node(random_node)
        for node in all_neighbor_nodes:
            search_time = int(generate_search_time_of_node())
            while search_time <= 0:
                search_time = int(generate_search_time_of_node())
            edge_add_time = cycle + search_time
            if edge_add_time not in nodes_to_be_connected_by_time:
                nodes_to_be_connected_by_time[edge_add_time] = [node,]
            else:
                nodes_to_be_connected_by_time[edge_add_time].append(node)
        most_degree_node = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][0]
        if cycle in nodes_to_be_connected_by_time:
            if connection_type == 'h':
                G.add_edges_from([(node,most_degree_node) for node in nodes_to_be_connected_by_time[cycle]])
            elif connection_type == 'r':
                for node in nodes_to_be_connected_by_time[cycle]:
                    G.add_edge(node,np.random.choice(G.nodes()))
        snapshots.append(G.copy())
    if generate_snapshot:
        return snapshots
    else:
        return G

# Calculate isolation for a graph G by gradually removing its nodes
def calculate_isolation(G):
    T = G.copy()

    is_isolated = any([len(c) == 1 for c in nx.connected_components(T)]) and len([c for c in nx.connected_components(T)]) > 1
    if is_isolated == 1:
        return 1

    return 0


m0 = int(input("Enter m0 (default: 3): ") or "3")
nmax = int(input("Enter nmax (default: 10): ") or "10")
cycles = int(input("Enter number of cycles (default: 100): ") or "100")
active = int(input("Enter active 0 or 1 (default: 1): ") or "1")
attacks = json.loads(input("Enter attacks as an array of cycle number (e.g: [30,50]): ") or "[]")
attacks_type = None
if len(attacks)>0:
    while True:
        attacks_type = input("Enter attacks type (h for highest degree, r for random): ") or "h"
        if attacks_type in ["h","r"]:
            break
        print("Invlid attacks type")
connection_type = None
if len(attacks)>0:
    while True:
        connection_type = input("Enter connection type for neighbors of attacked nodes (h for highest degree, r for random): ") or "h"
        if connection_type in ["h","r"]:
            break
        print("Invlid attacks type")
new_nodes_distribution = input("Select distribution for new nodes from poisson, binomial, exponential, pareto, degenerate, levy, cauchy & normal (default: poisson): ") or "poisson"
new_edges_distribution = input("Select distribution for new edges from exponential, uniform, constant & normal (default: exponential): ") or "exponential"

""""
# Output 1 - Graph evolutuion as a GIF
print("Generating GIF...")
all_G = modified_barabasi_albert(
    m0=m0,
    nmax=nmax,
    cycles=cycles,
    attacks=attacks,
    attacks_type=attacks_type,
    connection_type=connection_type,
    generate_snapshot=True,
    new_nodes_distribution=new_nodes_distribution,
    new_edges_distribution=new_edges_distribution,
    active=active
)
save_as_gif(all_G)

display_graph(all_G[cycles-1])
"""
# Output 2 - Degree distribution in log-log with CI
print("Calculation degree distribution ...")
all_G = []
for trial in range(20):
    print("Trial",trial+1,"out of 20")
    G = modified_barabasi_albert(
        m0=m0,
        nmax=nmax,
        cycles=cycles,
        attacks=attacks,
        attacks_type=attacks_type,
        connection_type=connection_type,
        generate_snapshot=False,
        new_nodes_distribution=new_nodes_distribution,
        new_edges_distribution=new_edges_distribution,
        active = active
    )
    all_G.append(G)
plot_degree_dist_with_ci(all_G)


# Output 3 - Isolation probability with CI
print("Calculation isolation probability ...")
isolation_probability = []
for trial in range(20):
    print("Trial",trial+1,"out of 20")
    all_G = modified_barabasi_albert(
        m0=m0,
        nmax=nmax,
        cycles=cycles,
        attacks=[],
        attacks_type=None,
        connection_type=connection_type,
        generate_snapshot=True,
        new_nodes_distribution=new_nodes_distribution,
        new_edges_distribution=new_edges_distribution,
        active = active
    )
    all_result = []

    for G in all_G:
        result = calculate_isolation(G)
        all_result.append(result)

    isolation_probability.append(all_result)

y_mean = np.average(isolation_probability,axis=0)
plot_isolation_probability_with_ci(y_mean,cycles)
