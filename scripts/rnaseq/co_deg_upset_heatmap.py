#!/usr/bin/env python3
"""
From *_DEG_classification.xlsx, produce:
  - UpSet plot (Core + Unique sheets)
  - Core-gene heatmap of log2FC (Condition columns)

Assumes sheets:
  - 'Core_Resilience'
  - 'Unique_<Label>' (e.g., Unique_Co_Drought / Unique_Co_Waterlogging / Unique_Co_Recovery)
"""
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from upsetplot import from_contents, UpSet

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--classification', required=True, help='Species DEG classification Excel')
    p.add_argument('--species', default='C. operculatus', help='For plot titles')
    p.add_argument('--out', default='out')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    core_df = pd.read_excel(args.classification, sheet_name='Core_Resilience')
    # collect all 'Unique_*' sheets present
    xls = pd.ExcelFile(args.classification)
    unique_sheets = [s for s in xls.sheet_names if s.startswith('Unique_')]
    unique = {s.replace('Unique_',''): pd.read_excel(args.classification, sheet_name=s)
              for s in unique_sheets}

    # UpSet
    gene_sets = {'Core': set(core_df['GeneID'])}
    for label, df in unique.items():
        gene_sets[label.split('_')[-1]] = set(df['GeneID'])  # label nice: Drought/Waterlogging/Recovery
    upset_data = from_contents(gene_sets)

    plt.figure(figsize=(10, 6))
    UpSet(upset_data, subset_size='count', show_counts='%d').plot()
    plt.suptitle(f'UpSet: Core and Unique DEGs — {args.species}')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, f'UpSet_{args.species.replace(\" \",\"_\")}.png'), dpi=300)
    plt.close('all')

    # Heatmap of core genes (log2FC by Condition)
    heatmap_df = core_df.pivot(index='GeneID', columns='Condition', values='log2FC').fillna(0)
    n_genes, n_conds = heatmap_df.shape
    plt.figure(figsize=(max(8, n_conds*2.2), max(6, n_genes*0.6)))
    sns.heatmap(heatmap_df, cmap='coolwarm', center=0, linewidths=0.4, linecolor='gray',
                square=True, cbar_kws={'label':'log2FC'})
    plt.title(f'Core Gene Expression (log2FC) — {args.species}')
    plt.ylabel('Gene ID'); plt.xlabel('Condition')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, f'Heatmap_Core_log2FC_{args.species.replace(\" \",\"_\")}.png'), dpi=300)
    plt.close('all')

if __name__ == '__main__':
    main()
