# Code created by M.Sabeeh Waqas
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/home/claude/owid-co2-data.csv')

# Filters countries for year 2022
year = 2022
cols = ['country', 'co2_per_capita', 'energy_per_capita',
        'gdp', 'coal_co2_per_capita']

iso_codes = df['iso_code'].dropna().unique()
df_yr = df[(df['year'] == year) & (df['iso_code'].notna())].copy()
df_yr = df_yr[cols + ['iso_code']].dropna()
df_yr = df_yr.drop_duplicates('country')

# Keep reasonable sample (top 40 by population coverage + well known)
np.random.seed(42)
df_plot = df_yr.copy().reset_index(drop=True)
df_plot = df_plot.head(50)  # take first 50 after dropna ordering

labels = df_plot['country'].tolist()
X = df_plot[['co2_per_capita', 'energy_per_capita', 'gdp', 'coal_co2_per_capita']].values

# Colour by co2_per_capita quintile
q = pd.qcut(df_plot['co2_per_capita'], 5, labels=['Very Low','Low','Medium','High','Very High'])
palette = ['#2166ac','#92c5de','#f7f7f7','#f4a582','#d6604d']
colors = [palette[['Very Low','Low','Medium','High','Very High'].index(v)] for v in q]
legend_patches = [mpatches.Patch(color=palette[i], label=l)
                  for i,l in enumerate(['Very Low','Low','Medium','High','Very High'])]

def plot_2d(ax, coords, labels, colors, title):
    ax.scatter(coords[:,0], coords[:,1], c=colors, s=60, alpha=0.85, edgecolors='k', linewidths=0.4)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (coords[i,0], coords[i,1]),
                    fontsize=5.5, xytext=(3,3), textcoords='offset points', alpha=0.8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel('Component 1', fontsize=7)
    ax.set_ylabel('Component 2', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(handles=legend_patches, fontsize=5, title='CO₂/capita', title_fontsize=5, loc='upper right')
    ax.grid(True, alpha=0.3, linewidth=0.5)

#  PCA 
# 1a. PCA without standardization
pca_raw = PCA(n_components=2, random_state=0)
X_pca_raw = pca_raw.fit_transform(X)

# 1b. PCA with standardization
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
pca_std = PCA(n_components=2, random_state=0)
X_pca_std = pca_std.fit_transform(X_std)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plot_2d(axes[0], X_pca_raw, labels, colors, 'PCA – Without Standardization')
plot_2d(axes[1], X_pca_std, labels, colors, 'PCA – With Standardization')
fig.suptitle('PCA: Climate Action (SDG 13) – Country Emissions Data (2022)',
             fontsize=10, fontweight='bold', y=1.01)
fig.tight_layout()
plt.savefig('/home/claude/fig_pca.png', dpi=160, bbox_inches='tight')
plt.close()
print("PCA saved")

#  MDS 
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, seed, tag in zip(axes, [0, 42], ['Seed 0', 'Seed 42']):
    mds = MDS(n_components=2, random_state=seed, dissimilarity='euclidean', max_iter=500)
    X_mds = mds.fit_transform(X)
    plot_2d(ax, X_mds, labels, colors, f'MDS – {tag} (Non-standardized)')
fig.suptitle('MDS: Climate Action (SDG 13) – Country Emissions Data (2022)',
             fontsize=10, fontweight='bold', y=1.01)
fig.tight_layout()
plt.savefig('/home/claude/fig_mds.png', dpi=160, bbox_inches='tight')
plt.close()
print("MDS saved")

#  t-SNE 
perplexities = [5, 15, 30]
seeds = [0, 42]

fig, axes = plt.subplots(3, 2, figsize=(13, 16))
for row, perp in enumerate(perplexities):
    for col, seed in enumerate(seeds):
        tsne = TSNE(n_components=2, perplexity=perp, random_state=seed,
                    max_iter=1000, init='random')
        X_tsne = tsne.fit_transform(X)
        plot_2d(axes[row][col], X_tsne, labels, colors,
                f't-SNE – Perplexity={perp}, Seed={seed}')

fig.suptitle('t-SNE: Climate Action (SDG 13) – Country Emissions Data (2022)',
             fontsize=11, fontweight='bold')
fig.tight_layout()
plt.savefig('/home/claude/fig_tsne.png', dpi=160, bbox_inches='tight')
plt.close()
print("t-SNE saved")

# Print data summary
print(f"\nDataset: {len(df_plot)} countries × 4 variables")
print("Variables: co2_per_capita, energy_per_capita, gdp, coal_co2_per_capita")
print("Year: 2022")
print(f"PCA (raw) explained var: {pca_raw.explained_variance_ratio_}")
print(f"PCA (std) explained var: {pca_std.explained_variance_ratio_}")
