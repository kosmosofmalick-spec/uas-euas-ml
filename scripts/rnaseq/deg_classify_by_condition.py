#!/usr/bin/env python3
"""
Classify DEGs per condition and export core/unique tables.

Inputs: one or more --input arguments in the form LABEL=PATH.xlsx
  where LABEL looks like: Co_Drought, Co_Waterlogging, Co_Recovery,
                          Sc_Drought, Sc_Waterlogging, Sc_Recovery

Required columns in each file:
  - '#ID' or 'GeneID'
  - 'log2FC', 'FDR', 'regulated'
  - 'COG_class', 'KOG_class', 'eggNOG_class'

Outputs (to --out dir):
  - Combined_DEG_classification.xlsx
  - C_operculatus_DEG_classification.xlsx (Co present)
  - S_cumini_DEG_classification.xlsx (Sc present)
"""
import argparse, os
import pandas as pd

def classify_function(row):
    cog = row.get('COG_class')
    kog = row.get('KOG_class')
    egg = row.get('eggNOG_class')
    if pd.notna(cog) and cog != '--':
        return f"{cog}_COG"
    if pd.notna(kog) and kog != '--':
        return f"{kog}_KOG"
    if pd.notna(egg) and egg != '--':
        return f"{egg}_eggNOG"
    return "Unclassified"

def load_and_filter(label, path, log2fc_thr, fdr_thr):
    df = pd.read_excel(path)
    if '#ID' in df.columns: df = df.rename(columns={'#ID': 'GeneID'})
    needed = {'GeneID','log2FC','FDR','regulated','COG_class','KOG_class','eggNOG_class'}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {missing} in {path}")
    degs = df[(df['log2FC'].abs() > log2fc_thr) & (df['FDR'] < fdr_thr)].copy()
    degs['Assigned_Class'] = degs.apply(classify_function, axis=1)
    sp, cond = label.split('_', 1)
    degs['Species'] = sp
    degs['Condition'] = cond
    return degs[['GeneID','Species','Condition','log2FC','FDR','regulated','Assigned_Class']]

def save_classification(gene_sets_subset, data_subset, out_path):
    # Core genes = intersection across labels in this subset
    core_genes = set.intersection(*gene_sets_subset.values()) if gene_sets_subset else set()
    unique_sets = {
        label: genes - set.union(*(gene_sets_subset[o] for o in gene_sets_subset if o != label))
        for label, genes in gene_sets_subset.items()
    }
    with pd.ExcelWriter(out_path) as xw:
        if core_genes:
            core_entries = [data_subset[label].loc[data_subset[label]['GeneID'].isin(core_genes)]
                            for label in gene_sets_subset]
            pd.concat(core_entries, ignore_index=True).to_excel(xw, sheet_name='Core_Resilience', index=False)
        for label, ids in unique_sets.items():
            u = data_subset[label].loc[data_subset[label]['GeneID'].isin(ids)]
            u.to_excel(xw, sheet_name=f'Unique_{label}', index=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', action='append', required=True,
                   help='LABEL=PATH.xlsx (e.g., Co_Drought=./Co_drought.xlsx)')
    p.add_argument('--log2fc', type=float, default=1.0)
    p.add_argument('--fdr', type=float, default=0.05)
    p.add_argument('--out', default='out')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    gene_sets, data = {}, {}
    for item in args.input:
        if '=' not in item:
            raise ValueError(f"Bad --input '{item}'. Use LABEL=PATH.")
        label, path = item.split('=', 1)
        d = load_and_filter(label.strip(), path.strip(), args.log2fc, args.fdr)
        data[label] = d
        gene_sets[label] = set(d['GeneID'])

    # Combined
    save_classification(gene_sets, data, os.path.join(args.out, 'Combined_DEG_classification.xlsx'))

    # Per species if present
    label_map = {'Co': 'C_operculatus', 'Sc': 'S_cumini'}
    for sp_prefix, sp_name in label_map.items():
        subset_labels = {k: v for k, v in gene_sets.items() if k.startswith(sp_prefix)}
        if subset_labels:
            save_classification(subset_labels, {k: data[k] for k in subset_labels},
                                os.path.join(args.out, f'{sp_name}_DEG_classification.xlsx'))

if __name__ == '__main__':
    main()


