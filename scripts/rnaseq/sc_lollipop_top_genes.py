#!/usr/bin/env python3
"""
Create lollipop plots of top/bottom genes per response category.

Input Excel (from classify_patterns_wl_vs_recovery.py) must contain:
  - 'GeneID', 'Category', 'log2FC_WL', 'log2FC_REC'

Categories expected: 'Persistent', 'Transient', 'Late-Response'
"""
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = {'Persistent': '#1f78b4', 'Transient': '#33a02c', 'Late-Response': '#e31a1c'}

def top_bottom(df, category, fc_col, n=5):
    sub = df[(df['Category']==category) & (~df[fc_col].isna())].copy()
    if sub.empty: return pd.DataFrame(columns=df.columns.tolist()+['Used_Condition'])
    tb = pd.concat([sub.nlargest(n, fc_col), sub.nsmallest(n, fc_col)])
    tb['Used_Condition'] = fc_col
    return tb

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--classification', required=True,
                   help='Pattern classification Excel (e.g., Co_Gene_Pattern_Classification_with_Annotation.xlsx)')
    p.add_argument('--species', default='S. cumini')
    p.add_argument('--topn', type=int, default=5)
    p.add_argument('--out', default='out/Top10_Lollipop_ByCategory.png')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    df = pd.read_excel(args.classification)
    df['Category'] = df['Category'].astype(str).str.strip().str.title()

    highlight = pd.concat([
        top_bottom(df, 'Persistent', 'log2FC_WL', args.topn),
        top_bottom(df, 'Transient', 'log2FC_WL', args.topn),
        top_bottom(df, 'Late-Response', 'log2FC_REC', args.topn),
    ], ignore_index=True)
    if highlight.empty:
        raise SystemExit("No data to plot. Check categories and columns.")

    # reshape for clean plotting
    highlight['value'] = highlight.apply(lambda r: r[r['Used_Condition']], axis=1)

    sns.set(style="whitegrid", font_scale=0.9)
    g = sns.FacetGrid(highlight, col="Category", sharex=False, sharey=False, height=5, aspect=0.9)

    def draw(data, color=None, **kwargs):
        ax = plt.gca()
        for _, r in data.iterrows():
            ax.plot([0, r['value']], [r['GeneID'], r['GeneID']], color='gray', linewidth=1, zorder=1)
        sns.scatterplot(data=data, x='value', y='GeneID', hue='Category',
                        palette=PALETTE, s=90, edgecolor='black', linewidth=0.5, legend=False, ax=ax)

    g.map_dataframe(draw)
    g.set_titles("{col_name}")
    g.set_xlabels("log2FC"); g.set_ylabels("")
    for ax in g.axes.flat: ax.axvline(0, color='gray', linestyle='--', linewidth=1)

    plt.suptitle(f"Top {args.topn} Up & Down Genes per Category — {args.species}", fontsize=14, y=1.05)
    plt.tight_layout()
    g.savefig(args.out, dpi=300)
    plt.close('all')
    print(f"✅ Saved: {args.out}")

if __name__ == '__main__':
    main()
