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
