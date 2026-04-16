"""
eda_plots.py
------------
Generates all EDA visualizations for the food nutrition dataset.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

PALETTE = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

def run_eda(df, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    print("[EDA] Generating visualizations...")

    _plot_macro_distributions(df, plots_dir)
    _plot_correlation_heatmap(df, plots_dir)
    _plot_group_boxplots(df, plots_dir)
    _plot_caloric_categories(df, plots_dir)
    _plot_nutrition_density(df, plots_dir)
    _plot_vitamin_mineral_bars(df, plots_dir)
    _plot_protein_vs_calories(df, plots_dir)
    _plot_fat_breakdown(df, plots_dir)
    _plot_macro_pairplot(df, plots_dir)

    print("[EDA] All 9 plots saved.")


def _plot_macro_distributions(df, plots_dir):
    from data_loader_compat import MACRO_COLS
    macro_cols = [c for c in MACRO_COLS if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, col in enumerate(macro_cols):
        axes[i].hist(df[col].dropna(), bins=60, color=PALETTE[i % 5],
                     edgecolor='white', linewidth=0.3, alpha=0.85)
        axes[i].set_title(col, fontweight='bold')
        axes[i].set_xlabel('Value'); axes[i].set_ylabel('Count')
        axes[i].axvline(df[col].median(), color='red', linestyle='--', linewidth=1.2,
                        label=f'Median: {df[col].median():.1f}')
        axes[i].legend(fontsize=8)
    for j in range(len(macro_cols), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Macronutrient Distributions', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '01_macro_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _plot_correlation_heatmap(df, plots_dir):
    from data_loader_compat import NUMERIC_COLS
    num_cols = [c for c in NUMERIC_COLS if c in df.columns]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(18, 15))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, square=True, linewidths=0.3, ax=ax,
                annot_kws={'size': 6}, cbar_kws={'shrink': 0.8})
    ax.set_title('Nutrient Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '02_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _plot_group_boxplots(df, plots_dir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    cols = ['Caloric Value', 'Protein', 'Fat', 'Carbohydrates', 'Dietary Fiber', 'Sugars']
    for i, col in enumerate(cols):
        if col in df.columns:
            groups = [df[df['Group'] == g][col].dropna() for g in sorted(df['Group'].unique())]
            bplot = axes[i].boxplot(groups, patch_artist=True, notch=False, showfliers=False)
            for j, patch in enumerate(bplot['boxes']):
                patch.set_facecolor(PALETTE[j % 5])
                patch.set_alpha(0.7)
            axes[i].set_xticklabels(sorted(df['Group'].unique()), rotation=15, fontsize=8)
            axes[i].set_title(col, fontweight='bold')
            axes[i].set_ylabel('Value')
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Nutrient Distribution by Food Group', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '03_group_boxplots.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _plot_caloric_categories(df, plots_dir):
    if 'Caloric Category' not in df.columns:
        return
    counts = df['Caloric Category'].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']
    bars = ax1.bar(counts.index, counts.values, color=colors[:len(counts)], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(val), ha='center', fontweight='bold')
    ax1.set_title('Food Count per Caloric Category', fontweight='bold')
    ax1.set_xlabel('Caloric Category'); ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=20)

    wedges, texts, autotexts = ax2.pie(
        counts.values, labels=counts.index, colors=colors[:len(counts)],
        autopct='%1.1f%%', startangle=90, pctdistance=0.85,
        wedgeprops=dict(edgecolor='white', linewidth=1.5))
    for at in autotexts:
        at.set_fontsize(9)
    ax2.set_title('Caloric Category Distribution', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '04_caloric_categories.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _plot_nutrition_density(df, plots_dir):
    if 'Nutrition Density' not in df.columns:
        return
    top = df.nlargest(25, 'Nutrition Density')[['food', 'Nutrition Density']].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top)))
    bars = ax.barh(top['food'], top['Nutrition Density'], color=colors, edgecolor='white')
    ax.set_xlabel('Nutrition Density Score', fontsize=11)
    ax.set_title('Top 25 Most Nutritionally Dense Foods', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '05_nutrition_density_ranking.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _plot_vitamin_mineral_bars(df, plots_dir):
    from data_loader_compat import VITAMIN_COLS, MINERAL_COLS
    vcols = [c for c in VITAMIN_COLS if c in df.columns]
    mcols = [c for c in MINERAL_COLS if c in df.columns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    vmeans = df[vcols].mean().sort_values(ascending=True)
    ax1.barh(vmeans.index, vmeans.values, color='#9C27B0', alpha=0.8, edgecolor='white')
    ax1.set_title('Average Vitamin Content', fontweight='bold')
    ax1.set_xlabel('Mean per 100g')

    mmeans = df[mcols].mean().sort_values(ascending=True)
    ax2.barh(mmeans.index, mmeans.values, color='#FF9800', alpha=0.8, edgecolor='white')
    ax2.set_title('Average Mineral Content', fontweight='bold')
    ax2.set_xlabel('Mean per 100g')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '06_vitamin_mineral_bars.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _plot_protein_vs_calories(df, plots_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    groups = sorted(df['Group'].unique())
    for i, grp in enumerate(groups):
        sub = df[df['Group'] == grp]
        ax.scatter(sub['Caloric Value'], sub['Protein'], alpha=0.5, s=18,
                   color=PALETTE[i % 5], label=grp)
    ax.set_xlabel('Caloric Value (kcal)', fontsize=11)
    ax.set_ylabel('Protein (g)', fontsize=11)
    ax.set_title('Protein vs Caloric Value by Food Group', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '07_protein_vs_calories.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _plot_fat_breakdown(df, plots_dir):
    fat_cols = ['Saturated Fats', 'Monounsaturated Fats', 'Polyunsaturated Fats']
    fat_cols = [c for c in fat_cols if c in df.columns]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fat_colors = ['#E91E63', '#FF9800', '#4CAF50']
    for i, col in enumerate(fat_cols):
        axes[i].hist(df[col].dropna(), bins=50, color=fat_colors[i],
                     edgecolor='white', alpha=0.85)
        axes[i].set_title(col, fontweight='bold')
        axes[i].set_xlabel('g per 100g'); axes[i].set_ylabel('Count')
        axes[i].axvline(df[col].median(), color='black', linestyle='--', linewidth=1.5,
                        label=f'Median: {df[col].median():.2f}')
        axes[i].legend(fontsize=8)
    plt.suptitle('Fat Type Distributions', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '08_fat_breakdown.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _plot_macro_pairplot(df, plots_dir):
    macro = ['Caloric Value', 'Protein', 'Fat', 'Carbohydrates', 'Dietary Fiber']
    macro = [c for c in macro if c in df.columns]
    sample = df[macro + ['Group']].dropna().sample(min(800, len(df)), random_state=42)

    palette_dict = {g: PALETTE[i % 5] for i, g in enumerate(sorted(sample['Group'].unique()))}
    g = sns.pairplot(sample, hue='Group', vars=macro, palette=palette_dict,
                     plot_kws={'alpha': 0.35, 's': 12}, diag_kind='kde')
    g.fig.suptitle('Macro Nutrient Pair Plot by Group', fontsize=13, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(plots_dir, '09_macro_pairplot.png'), dpi=150, bbox_inches='tight')
    plt.close()
