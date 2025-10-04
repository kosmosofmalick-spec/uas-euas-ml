# === Paste your file paths here ===
file_path_co = "C:/Users/Morpho graph 1.xlsx"
file_path_sc = "C:/Users/Morpho graph 2.xlsx"

# === Script starts here ===
import pandas as pd
import matplotlib.pyplot as plt

# Load the two datasets
df_co = pd.read_excel(file_path_co, sheet_name='Feuil1')
df_sc = pd.read_excel(file_path_sc, sheet_name='Feuil1')

# Add species names
df_co['Species'] = 'C. operculatus'
df_sc['Species'] = 'S. cumini'

# Calculate Total Root Fresh Weight (Primary + Adventitious)
df_co['Total_Root_FW'] = df_co['PRFW'] + df_co['ARFW']
df_sc['Total_Root_FW'] = df_sc['PRFW'] + df_sc['ARFW']

# Select only necessary columns
df_co_simple = df_co[['Treatments', 'Species', 'Total_Root_FW', 'SFW', 'LFW', 'TPFW']]
df_sc_simple = df_sc[['Treatments', 'Species', 'Total_Root_FW', 'SFW', 'LFW', 'TPFW']]

# Merge datasets
df_merged = pd.concat([df_co_simple, df_sc_simple], ignore_index=True)

# Group by Treatment and Species and calculate mean
means_combined = df_merged.groupby(['Treatments', 'Species'])[['Total_Root_FW', 'SFW', 'LFW', 'TPFW']].mean().reset_index()

# === Plotting starts here ===

# Colors
root_color = 'saddlebrown'   # brown for roots
stem_color = 'olivedrab'      # woody green for stems
leaf_color = 'limegreen'      # bright green for leaves

# Treatment order
treatments = ['CK', 'WL', 'RWL', 'D']

# Adjust x-axis positions
x_positions = []
for i in range(len(treatments)):
    x_positions.append(i * 3 + 0.2)  # C. operculatus
    x_positions.append(i * 3 + 0.8)  # S. cumini

# Bar width
bar_width = 0.55

# Update plot style for Nature-like aesthetics
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 13
})

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Plot each bar
for i, treatment in enumerate(treatments):
    for j, species in enumerate(['C. operculatus', 'S. cumini']):
        data = means_combined[(means_combined['Treatments'] == treatment) & (means_combined['Species'] == species)]
        idx = i * 3 + j

        root = data['Total_Root_FW'].values[0]
        stem = data['SFW'].values[0]
        leaf = data['LFW'].values[0]

        ax.bar(idx, root, width=bar_width, label='Root FW' if (i==0 and j==0) else "", color=root_color, edgecolor='black')
        ax.bar(idx, stem, width=bar_width, bottom=root, label='Stem FW' if (i==0 and j==0) else "", color=stem_color, edgecolor='black')
        ax.bar(idx, leaf, width=bar_width, bottom=root+stem, label='Leaf FW' if (i==0 and j==0) else "", color=leaf_color, edgecolor='black')

# Axis labels and title
ax.set_ylabel('Fresh Weight (g)')
ax.set_xlabel('Treatments')
ax.set_title('Comparison of Fresh Weights\nC. operculatus vs S. cumini')

# X-ticks
main_positions = [(i * 3 + 0.5) for i in range(len(treatments))]
ax.set_xticks(main_positions)
ax.set_xticklabels(treatments)

# Add thin gridlines
ais_major = ax.yaxis.get_major_locator()
ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), title="Plant Parts", loc='upper right', frameon=False)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Annotate total TPFW
for i, treatment in enumerate(treatments):
    for j, species in enumerate(['C. operculatus', 'S. cumini']):
        data = means_combined[(means_combined['Treatments'] == treatment) & (means_combined['Species'] == species)]
        idx = i * 3 + j
        total = data['TPFW'].values[0]
        ax.text(idx, total + 5, f"{total:.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()
