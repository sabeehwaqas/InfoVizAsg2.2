# Code created by M.Sabeeh Waqas
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd

# SDG edge list 
# Edges represent perceived interconnections between the 17 SDGs.
# My SDG is 13 
edges = [
    # Climate action (13)
    (13,  7), (13,  9), (13, 11), (13, 12), (13, 14), (13, 15), (13,  1), (13,  2), (13,  3), (13,  6),
    # Poverty and basic needs cluster
    ( 1,  2), ( 1,  3), ( 1,  8), ( 1, 10),( 2,  3), ( 2,  6), ( 2, 12), ( 2, 15),
    # Health and wellbeing
    ( 3,  6), ( 3, 10),
    # Education
    ( 4,  5), ( 4,  8), ( 4, 10), ( 4, 16),
    # Energy and Industry
    ( 7,  9), ( 7, 11),
    # Cities and consumption
    (11, 12), (12, 14),
    # Oceans and land
    (14, 15),
    # Partnerships
    (17, 13), (17,  1), (17,  9),
]
# De-duplicate
edges = list(set(tuple(sorted(e)) for e in edges))
print(f"Total edges: {len(edges)}")

# SDG names short 
sdg_names = {
    1:'No Poverty', 2:'Zero Hunger', 3:'Good Health', 4:'Quality Education',
    5:'Gender Equality', 6:'Clean Water', 7:'Clean Energy', 8:'Decent Work',
    9:'Industry & Innovation', 10:'Reduced Inequalities', 11:'Sustainable Cities',
    12:'Responsible Consumption', 13:'Climate Action', 14:'Life Below Water',
    15:'Life on Land', 16:'Peace & Justice', 17:'Partnerships'
}
# Official SDG colours , used online color picker
sdg_colors = {
    1:'#E5243B', 2:'#DDA63A', 3:'#4C9F38', 4:'#C5192D',
    5:'#FF3A21', 6:'#26BDE2', 7:'#FCC30B', 8:'#A21942',
    9:'#FD6925', 10:'#DD1367', 11:'#FD9D24', 12:'#BF8B2E',
    13:'#3F7E44', 14:'#0A97D9', 15:'#56C02B', 16:'#00689D',
    17:'#19486A'
}

G = nx.Graph()
G.add_nodes_from(range(1, 18))
G.add_edges_from(edges)

node_colors = [sdg_colors[n] for n in G.nodes()]
node_labels = {n: str(n) for n in G.nodes()}

def draw_graph(ax, pos, title):
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.45, width=1.2,
                           edge_color='#555555')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=520, alpha=0.95)
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax,
                            font_size=7, font_color='white', font_weight='bold')
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.axis('off')

#  1. Radial layout
fig, ax = plt.subplots(figsize=(8, 8))
# Numerical ordering on circle
theta = {n: 2 * np.pi * (n - 1) / 17 for n in range(1, 18)}
pos_radial = {n: (np.cos(theta[n]), np.sin(theta[n])) for n in range(1, 18)}
draw_graph(ax, pos_radial, 'SDG Network – Radial Layout (Numerical Node Ordering)')
# Add node name annotations outside
for n, (x, y) in pos_radial.items():
    ax.text(x * 1.18, y * 1.18, sdg_names[n], fontsize=5.5,
            ha='center', va='center', alpha=0.8, wrap=True)
fig.tight_layout()
plt.savefig('/home/claude/fig_network_radial.png', dpi=160, bbox_inches='tight')
plt.close()
print("Radial saved")

#  2. Kamada-Kawai with 2 seeds
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, seed in zip(axes, [0, 42]):
    np.random.seed(seed)
    pos_kk = nx.kamada_kawai_layout(G)
    draw_graph(ax, pos_kk, f'SDG Network – Kamada-Kawai (Seed {seed})')
fig.suptitle('Kamada-Kawai Network Layout – SDG Interconnections',
             fontsize=10, fontweight='bold')
fig.tight_layout()
plt.savefig('/home/claude/fig_network_kk.png', dpi=160, bbox_inches='tight')
plt.close()
print("KK saved")

#  3. Fruchterman-Reingold with 2 seeds
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, seed in zip(axes, [0, 42]):
    pos_fr = nx.spring_layout(G, seed=seed, k=1.5)
    draw_graph(ax, pos_fr, f'SDG Network – Fruchterman-Reingold (Seed {seed})')
fig.suptitle('Fruchterman-Reingold Network Layout – SDG Interconnections',
             fontsize=10, fontweight='bold')
fig.tight_layout()
plt.savefig('/home/claude/fig_network_fr.png', dpi=160, bbox_inches='tight')
plt.close()
print("FR saved")

# Print edge list as table
df_edges = pd.DataFrame(edges, columns=['SDG A', 'SDG B'])
df_edges['SDG A Name'] = df_edges['SDG A'].map(sdg_names)
df_edges['SDG B Name'] = df_edges['SDG B'].map(sdg_names)
df_edges = df_edges.sort_values(['SDG A','SDG B'])
print("\nEdge list:")
print(df_edges.to_string(index=False))
print(f"\nDegree of node 13 (Climate Action): {G.degree(13)}")
