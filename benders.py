# %%
import networkx as nx
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

import random
from math import sqrt

# Hyperparameters:
EPS = 1e-2
CUT_THRESHOLD = 1.0 - EPS
random.seed(42)


def dist(x1, y1, x2, y2):
    return round(sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2), 3)


def greedy_solution(source, nodes, space):
    if len(nodes) <= 1: return [], 0

    # Connect the source to the closest node:
    min_target = None
    min_arc_weight = float('inf')
    for v in nodes:
        if v == source: continue
        aux = space(source, v)
        if aux < min_arc_weight:
            min_target = v
            min_arc_weight = aux
    node_activation = {source: 0, min_target: min_arc_weight}
    sol_edges = [(source, min_target)]
    makespan = min_arc_weight

    # Init the degree map (-1 are not yet border nodes and -2 saturated nodes):
    degree = {v: -1 for v in nodes}
    degree[source] = -2
    degree[min_target] = 0

    border = [min_target]
    while len(sol_edges) < len(nodes) - 1:
        min_arc = (None, None)
        min_makespan = float('inf')
        for u in border:
            for v in nodes:
                if v == source or v == u: continue
                if degree[v] == -1:  # not yet added
                    aux = node_activation[u] + space(u, v)
                    if aux < min_makespan:
                        min_arc = (u, v)
                        min_makespan = aux
        u, v = min_arc
        if u is None: break
        sol_edges.append((u, v))
        node_activation[v] = min_makespan
        makespan = max(makespan, min_makespan)
        degree[u] += 1
        if degree[u] == 2:
            border.remove(u)
        degree[v] = 0
        border.append(v)
    return sol_edges, makespan


# Create a random graph and visualize it:
n = 30
r = 0  # the root node
G = nx.complete_graph(n)
pos = {i: (random.random(), random.random()) for i in list(range(n))}
for i, j, D in G.edges(data=True):
    (x1, y1) = pos[i]
    (x2, y2) = pos[j]
    D['w'] = dist(x1, y1, x2, y2)

# Warm start:
greedy_edges, greedy_makespan = greedy_solution(r, G.nodes, lambda u, v: G.edges[u, v]['w'])

# %%
import gurobipy as gp
from gurobipy import GRB

# Basic FTP edges model:
model = gp.Model()
# model.Params.OutputFlag = 0

# Edge usage variables:
x = model.addVars(G.edges, name='x', vtype=GRB.BINARY)
for i, j in G.edges: x[j, i] = x[i, j]

# Depth variable:
d = model.addVar(lb=0, ub=greedy_makespan, name='d', vtype=GRB.CONTINUOUS)
model.update()

# Objective function:
model.setObjective(d, GRB.MINIMIZE)

# Each node (besides r) should have a degree at most three:
model.addConstrs((gp.quicksum(x[i, j] for j in G.neighbors(i)) <= 3 for i in G.nodes if i != r), name='internal_d')

# The root has degree one:
model.addConstr(gp.quicksum(x[r, j] for j in G.neighbors(r)) == 1, name='root_d')

# The solution is a tree:
model.addConstr(gp.quicksum(x[i, j] for i, j in G.edges) == n - 1, name='total_edges')

model.update()
# %%
# Add path cover modeling:
f = model.addVars(G.edges, G.nodes, name='f', vtype=GRB.BINARY)
fk = {k: {(i, j): f[i, j, k] for i, j in G.edges} for k in G.nodes}
for i, j in G.edges:
    for k in G.nodes:
        f[j, i, k] = f[i, j, k]
        fk[k][j, i] = fk[k][i, j]
model.update()

# Path usage constraints:
for i, j in G.edges:
    for k in G.nodes:
        model.addConstr(f[i, j, k] <= x[i, j])

# Path degree constraints:
for k in G.nodes:
    if k == r: continue

    # Each path should start at the root and end at its target k:
    model.addConstr(gp.quicksum(f[r, i, k] for i in G.neighbors(r)) == 1)
    model.addConstr(gp.quicksum(f[k, i, k] for i in G.neighbors(k)) == 1)
    for i in G.nodes:
        if i == r or i == k: continue
        model.addConstr(gp.quicksum(f[i, j, k] for j in G.neighbors(i)) <= 2)
    model.addConstr(gp.quicksum(f[i, j, k] for i, j in G.edges) <= n - 1)

# Paths to depth connection constraints:
for k in G.nodes:
    if k == r: continue
    model.addConstr(d >= gp.quicksum(f[i, j, k] * G.edges[i, j]['w'] for i, j in G.edges))


# %%
def subtour_cb(x_var, x_val, adder, target=None):
    # Build an undirected graph with selected edges:
    subG = nx.Graph()
    for i, j in G.edges:
        if x_val[i, j] > EPS:
            subG.add_edge(i, j, capacity=x_val[i, j])

    # Compute Gomory-Hu tree:
    added = 0
    if len(subG.nodes) > 0:
        gh_tree = nx.gomory_hu_tree(subG, capacity='capacity')

        # Check all cuts in the Gomory-Hu tree
        for u, v, w in gh_tree.edges(data='weight'):
            if w > CUT_THRESHOLD: continue

            # Remove the edge to get a cut
            gh_copy = gh_tree.copy()
            gh_copy.remove_edge(u, v)

            # Get the two components:
            components = list(nx.connected_components(gh_copy))

            # Check if the cut is violated:
            S = components[0] if r in components[0] else components[1]
            if target is not None and target in S: continue

            work = 0
            for comp in components:
                internal_edges = []
                cost = 0
                for i, j in G.subgraph(comp).edges:
                    internal_edges.append(x_var[i, j])
                    cost += x_val[i, j]
                work += cost > len(comp) - 1

                if cost > len(comp) - 1:
                    adder(gp.quicksum(internal_edges) <= len(comp) - 1)
                    added += 1

            # if work > 0: continue

            # Add lazy constraint: sum of edges leaving S >= 1
            cut_edges = []
            for i in S:
                for j in G.neighbors(i):
                    if j not in S:
                        cut_edges.append(x_var[i, j])

            if len(cut_edges) > 0:
                adder(gp.quicksum(cut_edges) >= 1)
                added += 1
    return added


# %%
def path_cb(f_val, adder):
    added = 0
    for k in G.nodes:
        if k == r: continue
        x_val = {(i, j): f_e_k for (i, j, l), f_e_k in f_val.items() if l == k}
        added += subtour_cb(fk[k], x_val, adder, target=k)
    return added


# %%
def cb(m, where):
    if where != GRB.Callback.MIPSOL and where != GRB.Callback.MIPNODE: return

    # Chose the function to get variable values depending on when the callback is called:
    if where == GRB.Callback.MIPNODE:
        if m.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL: return
        getval = lambda var: abs(m.cbGetNodeRel(var))
    else:
        getval = lambda var: abs(m.cbGetSolution(var))

    adder = m.cbCut if where == GRB.Callback.MIPNODE else m.cbLazy
    # Get current solution:
    x_val = {e: getval(x_e) for e, x_e in x.items()}
    f_val = {e_k: getval(f_e_k) for e_k, f_e_k in f.items()}

    subtour_cb(x, x_val, adder)
    path_cb(f_val, adder)


# %%
# Model solving:
model.update()

# Set x variables:
for i, j in G.edges:
    if (i, j) in greedy_edges or (j, i) in greedy_edges:
        x[i, j].Start = 1
    else:
        x[i, j].Start = 0
# Set d variable:
d.Start = greedy_makespan

# Set f variables:
tree = nx.Graph(greedy_edges)
for k in G.nodes:
    if k == r: continue
    path = nx.shortest_path(tree, r, k)
    path_edges = set(zip(path, path[1:]))
    for i, j in G.edges:
        if (i, j) in path_edges or (j, i) in path_edges:
            f[i, j, k].Start = 1
        else:
            f[i, j, k].Start = 0

model.Params.LazyConstraints = 1
model.optimize(cb)
# %%
# Visualize the solution:
sol_x = model.getAttr('x', x)
sol_f = model.getAttr('x', f)
sol_edges = [e for e in G.edges if sol_x[e] > 1.0 - EPS]
node_colors = []
colors = list(mcolors.BASE_COLORS)
if n <= 10:
    for k in G.nodes:
        if k == r:
            node_colors.append('gray')
            continue
        curr_color = colors[k % len(colors)]
        node_colors.append(curr_color)
        path = [(i, j) for i, j in G.edges if sol_f[i, j, k] > 1.0 - EPS]
        nx.draw_networkx_edges(G, pos=pos, edgelist=path, edge_color=curr_color, style='dashed',
                               connectionstyle=f'arc3,rad={(k + 1) * 0.1}', arrows=True)
else:
    node_colors = ['green' if i == r else 'gray' for i in G.nodes]
edge_color = ['black' if e in sol_edges else 'white' for e in G.edges]
nx.draw(G, pos=pos, with_labels=True, node_color=node_colors, edge_color=edge_color)
plt.show()

print(f'Makespan = {model.objVal:.3f}')
