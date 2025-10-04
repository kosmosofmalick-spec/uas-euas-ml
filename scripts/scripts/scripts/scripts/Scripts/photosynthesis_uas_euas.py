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
