# %%
import random
import networkx as nx
from matplotlib import pyplot as plt

from math import log2, sqrt, ceil

# Hyperparameters:
EPS = 1e-2
CUT_THRESHOLD = 0.3 - EPS
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
DG = nx.complete_graph(n, nx.DiGraph)
pos = {i: (random.random(), random.random()) for i in list(range(n))}
for i, j, D in DG.edges(data=True):
    (x1, y1) = pos[i]
    (x2, y2) = pos[j]
    D['w'] = dist(x1, y1, x2, y2)

root_dist = {}
rootx, rooty = pos[r]
for i, (x, y) in pos.items():
    root_dist[i] = dist(x, y, rootx, rooty)

max_edge = max(d['w'] for u, v, d in DG.edges(data=True))
M = ceil(log2(DG.number_of_nodes())) * max_edge

# Warm start:
greedy_edges, greedy_makespan = greedy_solution(r, DG.nodes, lambda u, v: DG.edges[u, v]['w'])
M = min(M, greedy_makespan)

# %%
import gurobipy as gp
from gurobipy import GRB

# Basic FTP edges model:
model = gp.Model()
# model.Params.OutputFlag = 0

# Edge usage variables:
x = model.addVars(DG.edges, name='x', vtype=GRB.BINARY)

# Depth variable:
D = model.addVar(lb=0, ub=M, vtype=GRB.CONTINUOUS)
model.update()

# Objective function:
model.setObjective(D, GRB.MINIMIZE)

# Each node (besides r) should have one incoming arc:
model.addConstrs(gp.quicksum(x[j, i] for j in DG.predecessors(i)) == 1 for i in DG.nodes if i != r)

# No incoming arc for the root:
model.addConstr(gp.quicksum(x[j, r] for j in DG.predecessors(r)) == 0)

# The root has out-degree one:
model.addConstr(gp.quicksum(x[r, j] for j in DG.successors(r)) == 1)

# Each node (besides r) should have at most two outgoing arcs:
model.addConstrs(gp.quicksum(x[i, j] for j in DG.successors(i)) <= 2 for i in DG.nodes if i != r)

# The solution is a tree:
model.addConstr(x.sum() == n - 1, name='total_edges')

# For each arc pair use at most one arc:
for i, j in DG.edges:
    model.addConstr(x[i, j] + x[j, i] <= 1)

model.update()
# %%
# Add depth variable modeling:
d = model.addVars(DG.nodes, lb=0, ub=M, name='d', vtype=GRB.CONTINUOUS)
model.update()

for j in DG.nodes:
    if j == r: continue
    for i in DG.predecessors(j):
        # A node's depth is at least the depth of its predecessor plus the distance between them:
        model.addConstr(d[j] >= d[i] + DG.edges[i, j]['w'] * x[i, j] + M * (x[i, j] - 1))

# The tree depth is the maximum over all depths:
for _, depth in d.items():
    model.addConstr(D >= depth)

for i, depth in d.items():
    depth.LB = root_dist[i]


# %%
def subtour_cb(x_val, adder, target=None):
    # Build an undirected graph with selected edges:
    G = nx.Graph()
    for i, j in DG.edges:
        if x_val[i, j] > EPS:
            G.add_edge(i, j)

    components = list(nx.connected_components(G))
    if len(components) <= 1: return 0
    for comp in components:
        internal_edges = []
        cost = 0
        for i, j in G.subgraph(comp).edges:
            internal_edges.append(x[i, j])
            internal_edges.append(x[j, i])
            cost += x_val[i, j] + x_val[j, i]
        if cost > len(comp) - 1 + EPS:
            adder(gp.quicksum(internal_edges) <= len(comp) - 1)


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

    subtour_cb(x_val, adder)


# %%
# Model solving:
model.update()

# Warm start:
greedy_edges, greedy_makespan = greedy_solution(r, DG.nodes, lambda u, v: DG.edges[u, v]['w'])

# Set x variables:
for i, j in DG.edges:
    if (i, j) in greedy_edges:
        x[i, j].Start = 1
    else:
        x[i, j].Start = 0

# Set d variable:
D.Start = greedy_makespan

model.Params.LazyConstraints = 1
model.Params.TimeLimit = 600
model.optimize(cb)
# %%
# Visualize the solution:
sol_x = model.getAttr('x', x)
sol_edges = [e for e in DG.edges if sol_x[e] > 1.0 - EPS]
node_colors = ['green' if i == r else 'gray' for i in DG.nodes]
nx.draw(DG.edge_subgraph(sol_edges), pos=pos, with_labels=True, node_color=node_colors)
plt.show()

print(f'Makespan = {model.objVal:.3f}')
