
## Repository tree

```
uas-euas-ml/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ env.yml
├─ .gitignore
├─ SEED.txt
├─ CONFIG.md
├─ data/
│  └─ README.md
├─ out/              # created at runtime (plots/models saved here)
└─ scripts/
   ├─ morphology_stacked_fw.py
   ├─ adventitious_root_cross_section.py
   ├─ rf_feature_importance.py
   └─ physiology_uas_euas.py
```

---

## README.md

````markdown
# UAS/EUAS & ML Pipeline for Forest Tree Stress

This repository contains cleaned, reproducible scripts (no personal names/paths) for:
- Morphology stacked fresh-weight plots across treatments/species
- Adventitious root cross‑section schematic (cortex/stele)
- Random‑forest feature importance on morphological traits
- UAS / Enhanced UAS (EUAS) from physiological traits

Processed data for the associated manuscript are available at the dataset DOI: **10.5281/zenodo.17261830**.

## Quick start
```bash
# 1) Create env (conda) or use requirements.txt with pip
conda env create -f env.yml
conda activate trees-ml

# 2) Run scripts (examples below)
python scripts/morphology_stacked_fw.py \
  --co_excel data/morpho_co.xlsx --sc_excel data/morpho_sc.xlsx \
  --sheet Feuil1 --out out

python scripts/adventitious_root_cross_section.py \
  --co_excel data/anatomy_co.xlsx --sc_excel data/anatomy_sc.xlsx \
  --sheet Feuil1 --out out

python scripts/rf_feature_importance.py \
  --excel data/morphology_all.xlsx --label_col Treatments --out out

python scripts/physiology_uas_euas.py \
  --excel data/physiology.xlsx --out out
````

## Expected input formats (see `data/README.md`)

* **Morphology (per‑species files):** columns include `Treatments, PRFW, ARFW, SFW, LFW, TPFW`.
* **Anatomy (per‑species files):** columns include `Treatment, Cortex area, Stele area, Cross-sectional area, Cortex area/Stele area`.
* **RF features:** a single Excel with one sheet; column `Treatments` + numeric trait columns.
* **Physiology:** one Excel; columns follow pattern `Trait Co` and `Trait Sc` for each trait at each treatment; and a `Treatment` column containing at least `CK`.

## Reproducibility

* Random seed stored in `SEED.txt` (default `42`).
* All figures/models saved to `out/`.

## License

* **Code:** MIT (see `LICENSE`).
* **Data:** Cite external Zenodo dataset DOI above.

## Citation

If you use this code, please cite the dataset DOI above and your paper. If you archive a release of this repo on Zenodo, add that DOI here as well.

````

---

## LICENSE (MIT)
```text
MIT License

Copyright (c) 2025 The Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
````

---

## requirements.txt

```text
pandas
numpy
matplotlib
scikit-learn==1.2.2
xgboost==1.7.6
```

## env.yml

```yaml
name: trees-ml
channels: [conda-forge]
dependencies:
  - python=3.11
  - pandas
  - numpy
  - matplotlib
  - scikit-learn=1.2.2
  - xgboost=1.7.6
```

## .gitignore

```text
# Python
__pycache__/
*.pyc
.venv/
.env/

# OS/editor
.DS_Store
Thumbs.db

# Output
out/
```

## SEED.txt

```text
42
```

## CONFIG.md

```markdown
# Configuration
- Random seed: 42 (see SEED.txt)
- Output directory: `out/`
- Default Excel sheet: `Feuil1`
- UAS lambda (λ): 0.5 (physiology script)
```

## data/README.md

```markdown
# Data formats

## Morphology per-species files (for `morphology_stacked_fw.py`)
Required columns:
- Treatments (e.g., CK, WL, RWL, D)
- PRFW, ARFW (fresh weights)
- SFW (stem), LFW (leaf), TPFW (total plant FW)

## Anatomy per-species files (for `adventitious_root_cross_section.py`)
Required columns:
- Treatment (e.g., WL, RWL, WR)
- Cortex area, Stele area, Cross-sectional area, Cortex area/Stele area

## RF features (for `rf_feature_importance.py`)
- One Excel/CSV with `Treatments` column and numeric trait columns.

## Physiology (for `physiology_uas_euas.py`)
- One Excel/CSV with a column `Treatment` (includes `CK`).
- For each trait, two columns: `{Trait} Co` and `{Trait} Sc`.
```

---

## scripts/morphology_stacked_fw.py

```python
#!/usr/bin/env python3
"""
Stacked fresh-weight bars (root/stem/leaf) by Treatment × Species.
Inputs: two Excel files (per species) to avoid personal paths.
"""

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
```

---

## scripts/adventitious_root_cross_section.py

```python
#!/usr/bin/env python3
"""
Schematic cross-section circles for cortex/stele by Species × Treatment.
Areas are visualized with radii ~ sqrt(area). Saves a 1×4 panel.
"""
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
```

---

## scripts/rf_feature_importance.py

```python
#!/usr/bin/env python3
"""
Random-forest feature importance on morphology (name-scrubbed; no seaborn required).
Input Excel must have a column `Treatments` and numeric feature columns.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def main(args):
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_excel(args.excel)

    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found in Excel.")

    y = df[args.label_col]
    X = df.drop(columns=[args.label_col])

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    Xtr, Xte, ytr, yte = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    rf.fit(Xtr, ytr)
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(X.columns[order][::-1], importances[order][::-1])
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'rf_feature_importance.png'), dpi=300)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--excel', required=True)
    p.add_argument('--label_col', default='Treatments')
    p.add_argument('--out', default='out')
    main(p.parse_args())
```

---

## scripts/physiology_uas_euas.py

```python
#!/usr/bin/env python3
"""
Compute CR, UAS, and Enhanced UAS from physiology Excel.
Expects columns:
- Treatment (includes 'CK')
- For each trait T: 'T Co' and 'T Sc'
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your Excel file
data_path = "C:/Users/Physiological  1.xlsx"
df = pd.read_excel(data_path)

# Define traits and categories
traits = ['Soluble protein', 'Proline', 'SOD', 'POD', 'Superoxide anion', 'H2O2', 'MDA',
          'Relative conductivity', 'Leaf water potential', 'Relative leaf water content']

beneficial_traits = ['Soluble protein', 'Proline', 'SOD', 'POD', 'Relative leaf water content']
damage_traits = ['Superoxide anion', 'H2O2', 'MDA', 'Relative conductivity', 'Leaf water potential']

# Step 1: Compute means per treatment per species
means_df = df.groupby('Treatment').mean().reset_index()
control_means = means_df[means_df['Treatment'] == 'CK'].iloc[0]

# LRR calculator
def custom_lrr(value, control, trait):
    eps = 1e-9
    if trait in beneficial_traits:
        return np.log((value + eps) / (control + eps))
    else:
        return -np.log((value + eps) / (control + eps))

# Step 2–6: Compute CR, UAS, and Enhanced UAS
uas_records = []
trait_CR_matrix = {}

for _, row in means_df.iterrows():
    treatment = row['Treatment']
    CRs = []
    co_wins = 0
    sc_wins = 0

    for trait in traits:
        co_mean = row[f'{trait} Co']
        sc_mean = row[f'{trait} Sc']
        co_ck = control_means[f'{trait} Co']
        sc_ck = control_means[f'{trait} Sc']

        lrr_co = custom_lrr(co_mean, co_ck, trait)
        lrr_sc = custom_lrr(sc_mean, sc_ck, trait)
        cr = lrr_co - lrr_sc
        CRs.append(cr)

        if treatment != 'CK':
            if cr > 0:
                co_wins += 1
            elif cr < 0:
                sc_wins += 1

        if treatment not in trait_CR_matrix:
            trait_CR_matrix[treatment] = {}
        trait_CR_matrix[treatment][trait] = cr

    uas = np.mean(CRs)
    if treatment == 'CK':
        uas = np.mean([np.log(row[f'{t} Co'] / row[f'{t} Sc']) for t in traits])
        enhanced_uas = uas
    else:
        trait_adv = (co_wins - sc_wins) / len(traits)
        enhanced_uas = uas + 0.5 * trait_adv

    uas_records.append({
        'Treatment': treatment,
        'Original_UAS': uas,
        'Trait_Advantage': (co_wins - sc_wins),
        'Enhanced_UAS': enhanced_uas
    })

uas_df = pd.DataFrame(uas_records)

# Plot Step: Enhanced UAS
colors = ['#EF476F', '#118AB2', '#06D6A0', '#FFD166']
plt.figure(figsize=(10, 6))
bars = plt.bar(uas_df['Treatment'], uas_df['Enhanced_UAS'], color=colors, edgecolor='black')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3), ha='center', va='bottom', fontsize=11)

plt.axhline(0, color='black', linestyle='--')
plt.title('Enhanced Unified Advantage Score by Treatment\n(Physiological Parameters)', fontsize=14, fontweight='bold')
plt.ylabel('Enhanced UAS (Positive = C. operculatus advantage)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Plot Step: Trait-level CR for each treatment including CK
trait_CR_df = pd.DataFrame.from_dict(trait_CR_matrix, orient='index')

for treatment in trait_CR_df.index:
    cr_series = trait_CR_df.loc[treatment].sort_values()
    plt.figure(figsize=(10, 6))
    bars = plt.barh(cr_series.index, cr_series.values, color='skyblue', edgecolor='black')

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01 * np.sign(width), bar.get_y() + bar.get_height()/2,
                 f'{width:.2f}', va='center', ha='left' if width > 0 else 'right', fontsize=10)

    plt.axvline(0, color='black', linestyle='--')
    plt.title(f'Trait-Level Comparative Resilience (CR)\nTreatment: {treatment}', fontsize=14, fontweight='bold')
    plt.xlabel('CR = LRR_Co − LRR_Sc (Positive = C. operculatus advantage)', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
```

---

### Scripts/photosynthesis_uas_euas.py

```python
#!/usr/bin/env python3
"""
Enhanced UAS from photosynthetic traits (all beneficial).
Input Excel must have:
- A column `Treatment` (includes 'CK')
- Columns for each trait with species suffixes:
  'A C operculatus', 'A S cumini', 'gs C operculatus', 'gs S cumini', ...
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the photosynthetic dataset
file_path = "C:/Users/Photo 1.xlsx"
df = pd.read_excel(file_path)

# Define photosynthetic traits (all beneficial)
traits = ['A', 'gs', 'E', 'TChl', 'Fv/Fm']

# Calculate means per treatment
means_df = df.groupby('Treatment').mean().reset_index()
control_means = means_df[means_df['Treatment'] == 'CK'].iloc[0]

# LRR calculator for beneficial traits
def lrr(value, control):
    eps = 1e-9
    return np.log((value + eps) / (control + eps))

# Compute CR and Enhanced UAS
uas_records = []
trait_CR_matrix = {}

for _, row in means_df.iterrows():
    treatment = row['Treatment']
    CRs = []
    co_wins = 0
    sc_wins = 0

    for trait in traits:
        co_mean = row[f'{trait} C operculatus']
        sc_mean = row[f'{trait} S cumini']
        co_ck = control_means[f'{trait} C operculatus']
        sc_ck = control_means[f'{trait} S cumini']

        lrr_co = lrr(co_mean, co_ck)
        lrr_sc = lrr(sc_mean, sc_ck)
        cr = lrr_co - lrr_sc
        CRs.append(cr)

        if treatment != 'CK':
            if cr > 0:
                co_wins += 1
            elif cr < 0:
                sc_wins += 1

        if treatment not in trait_CR_matrix:
            trait_CR_matrix[treatment] = {}
        trait_CR_matrix[treatment][trait] = cr

    uas = np.mean(CRs)
    if treatment == 'CK':
        uas = np.mean([np.log(row[f'{t} C operculatus'] / row[f'{t} S cumini']) for t in traits])
        enhanced_uas = uas
    else:
        trait_adv = (co_wins - sc_wins) / len(traits)
        enhanced_uas = uas + 0.5 * trait_adv

    uas_records.append({
        'Treatment': treatment,
        'Original_UAS': uas,
        'Trait_Advantage': co_wins - sc_wins,
        'Enhanced_UAS': enhanced_uas
    })

uas_df = pd.DataFrame(uas_records)

# Plot Enhanced UAS
colors = ['#FF6F61', '#6B5B95', '#88B04B', '#FFA07A']
plt.figure(figsize=(10, 6))
bars = plt.bar(uas_df['Treatment'], uas_df['Enhanced_UAS'], color=colors, edgecolor='black')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3), ha='center', va='bottom', fontsize=11)

plt.axhline(0, color='black', linestyle='--')
plt.title('Enhanced Unified Advantage Score by Treatment\n(Photosynthetic Parameters)', fontsize=14, fontweight='bold')
plt.ylabel('Enhanced UAS (Positive = C. operculatus advantage)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Trait-level CR plots
trait_CR_df = pd.DataFrame.from_dict(trait_CR_matrix, orient='index')

for treatment in trait_CR_df.index:
    cr_series = trait_CR_df.loc[treatment].sort_values()
    plt.figure(figsize=(10, 6))
    bars = plt.barh(cr_series.index, cr_series.values, color='lightblue', edgecolor='black')

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01 * np.sign(width), bar.get_y() + bar.get_height()/2,
                 f'{width:.2f}', va='center', ha='left' if width > 0 else 'right', fontsize=10)

    plt.axvline(0, color='black', linestyle='--')
    plt.title(f'Trait-Level Comparative Resilience (CR)\nTreatment: {treatment}', fontsize=14, fontweight='bold')
    plt.xlabel('CR = LRR_Co − LRR_Sc (Positive = C. operculatus advantage)', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

```

---

### README additions (usage)

```markdown
python scripts/photosynthesis_uas_euas.py --excel data/photosynthesis.xlsx --out out
```

### data/README additions

```markdown
## Photosynthesis (for `photosynthesis_uas_euas.py`)
- One Excel/CSV with a `Treatment` column (includes `CK`).
- For each trait (default: A, gs, E, TChl, Fv/Fm) two columns named:
  `A C operculatus`, `A S cumini`, `gs C operculatus`, `gs S cumini`, `E C operculatus`, `E S cumini`, `TChl C operculatus`, `TChl S cumini`, `Fv/Fm C operculatus`, `Fv/Fm S cumini`.

```
