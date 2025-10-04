# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

# === Paste your file paths here ===
file_path_co_root = "C:/Users/C operculatus anatomy of AR.xlsx"
file_path_sc_root = "C:/Users/S cumini anatomy of AR.xlsx"

# === Load the datasets ===
df_co_root = pd.read_excel(file_path_co_root, sheet_name='Feuil1')
df_sc_root = pd.read_excel(file_path_sc_root, sheet_name='Feuil1')

# Add species names
df_co_root['Species'] = 'C. operculatus'
df_sc_root['Species'] = 'S. cumini'

# Correct column name if needed
df_co_root = df_co_root.rename(columns={'Treatements': 'Treatment'})

# Merge both datasets
df_root_combined = pd.concat([df_co_root, df_sc_root], ignore_index=True)

# Calculate means for each Treatment and Species
means_root = df_root_combined.groupby(['Treatment', 'Species'])[['Cortex area', 'Stele area', 'Cross-sectional area', 'Cortex area/Stele area']].mean().reset_index()

# Define the order for plotting
species_order = [
    ('C. operculatus', 'WL'),
    ('C. operculatus', 'RWL'),
    ('S. cumini', 'WL'),
    ('S. cumini', 'WR')
]

# Define colors
cortex_color = 'lightgreen'
stele_color = 'forestgreen'

# Create figure
fig, axes = plt.subplots(1, 4, figsize=(22, 7))

# Plot settings
for ax, (species, treatment) in zip(axes, species_order):
    data = means_root[(means_root['Species'] == species) & (means_root['Treatment'] == treatment)]
    if not data.empty:
        cortex_area = data['Cortex area'].values[0]
        stele_area = data['Stele area'].values[0]
        cross_area = data['Cross-sectional area'].values[0]

        # Normalize sizes: radius ~ sqrt(area)
        total_radius = (cross_area)**0.5
        stele_radius = (stele_area)**0.5

        # Draw outer circle (Cortex)
        outer_circle = plt.Circle((0, 0), total_radius, color=cortex_color, ec='black')
        ax.add_artist(outer_circle)

        # Draw inner circle (Stele)
        stele_circle = plt.Circle((0, 0), stele_radius, color=stele_color, ec='black')
        ax.add_artist(stele_circle)

        # Basic settings
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{species}\n{treatment}', fontsize=12, fontweight='bold')

        # Add scalebar
        ax.plot([-1.5, -0.5], [-2.2, -2.2], color='black', linewidth=2)
        ax.text(-1.0, -2.4, '1 mm', ha='center', fontsize=10)

# Global figure title
plt.suptitle('Adventitious Root Cross-Section Anatomy\nC. operculatus vs S. cumini', fontsize=18, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
