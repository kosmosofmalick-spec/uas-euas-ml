#!/usr/bin/env python3
"""
WL vs REC DEG pattern classification for a species.

Inputs:
  - --wl  path to Waterlogging Excel
  - --rec path to Recovery Excel

Required columns: '#ID' or 'GeneID', 'log2FC', 'FDR', 'regulated',
                  'COG_class','KOG_class','eggNOG_class'
"""
import argparse, os
import pandas as pd

def assign_class(row):
    cog = row.get('COG_class'); kog = row.get('KOG_class'); egg = row.get('eggNOG_class')
    if pd.notna(cog) and cog != '--':   return f"R_{cog}_COG"
    if pd.notna(kog) and kog != '--':   return f"R_{kog}_KOG"
    if pd.notna(egg) and egg != '--':   return f"R_{egg}_eggNOG"
    return 'Unclassified'

def std_geneid(df):
    return df.rename(columns={'#ID': 'GeneID'}) if '#ID' in df.columns else df

def classify_row(row):
    wl = pd.notna(row['log2FC_WL']); rc = pd.notna(row['log2FC_REC'])
    if wl and rc and (row['regulated_WL'] == row['regulated_REC']):
        return 'Persistent'
    if wl and not rc:
        return 'Transient'
    if rc and not wl:
        return 'Late-Response'
    return 'Other'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wl', required=True, help='Co_WL Excel path')
    ap.add_argument('--rec', required=True, help='Co_REC Excel path')
    ap.add_argument('--log2fc', type=float, default=1.0)
    ap.add_argument('--fdr', type=float, default=0.05)
    ap.add_argument('--out', default='out/Co_Gene_Pattern_Classification_with_Annotation.xlsx')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    wl_df = std_geneid(pd.read_excel(args.wl))
    rec_df = std_geneid(pd.read_excel(args.rec))

    wl = wl_df[(wl_df['FDR'] < args.fdr) & (wl_df['log2FC'].abs() > args.log2fc)].copy()
    rc = rec_df[(rec_df['FDR'] < args.fdr) & (rec_df['log2FC'].abs() > args.log2fc)].copy()

    annot_cols = ['GeneID','COG_class','KOG_class','eggNOG_class']
    wl_ann = wl_df[annot_cols].drop_duplicates()
    rc_ann = rec_df[annot_cols].drop_duplicates()
    annotations = pd.concat([wl_ann, rc_ann]).drop_duplicates('GeneID')
    annotations['Assigned_Class'] = annotations.apply(assign_class, axis=1)

    merged = pd.merge(wl[['GeneID','log2FC','regulated']],
                      rc[['GeneID','log2FC','regulated']],
                      on='GeneID', how='outer', suffixes=('_WL','_REC'))
    merged = pd.merge(merged, annotations[['GeneID','Assigned_Class']], on='GeneID', how='left')
    merged['Category'] = merged.apply(classify_row, axis=1)

    merged.to_excel(args.out, index=False)
    print(f"Saved: {args.out}")

if __name__ == '__main__':
    main()
