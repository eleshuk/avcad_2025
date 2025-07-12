# cover_crop_plots.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="whitegrid")


def boxplot_soc(df):
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=df,
        x="CoverCropGroup",
        y="BackgroundSOC",
        hue="CoverCropGroup",
        palette="Set3",
        legend=False,
        ax=ax
    )
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel("Cover Crop Group")
    ax.set_ylabel("Soil Organic Carbon (%)")
    ax.set_title("Distribution of Soil Organic Carbon by Cover Crop Group")
    plt.tight_layout()
    return fig

def barplot_country_freq(df):
    df = df.copy()
    df["Country"] = df["Country"].astype(str).str.strip().str.title()
    country_counts = df["Country"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=country_counts.head(10).values,
        y=country_counts.head(10).index,
        palette="viridis",
        ax=ax
    )
    ax.set_xlabel("Number of Observations")
    ax.set_ylabel("Country")
    ax.set_title("Top 10 Countries in the Dataset")
    plt.tight_layout()
    return fig

def heatmap_soc_means(df):
    df = df.copy()
    df["Country"] = df["Country"].astype(str).str.strip().str.title()
    top_countries = df["Country"].value_counts().head(6).index.tolist()
    df_top = df[df["Country"].isin(top_countries)]
    pivot_df = df_top.groupby(["CoverCropGroup", "Country"])["BackgroundSOC"].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot_df,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        linewidths=0.5,
        linecolor="gray",
        ax=ax
    )
    ax.set_title("Average Soil Organic Carbon (%) by Cover Crop Group and Top Countries")
    ax.set_ylabel("Cover Crop Group")
    ax.set_xlabel("Country")
    plt.tight_layout()
    return fig

def violinplot_soc_by_country(df):
    df = df.copy()
    df["Country"] = df["Country"].astype(str).str.strip().str.title()
    top_countries = df["Country"].value_counts().head(6).index.tolist()
    df_top = df[df["Country"].isin(top_countries)]
    group_counts = df_top.groupby(["Country", "CoverCropGroup"]).size().reset_index(name="n")
    valid_groups = group_counts[group_counts["n"] >= 3][["Country", "CoverCropGroup"]]
    df_filtered = df_top.merge(valid_groups, on=["Country", "CoverCropGroup"], how="inner")

    group_order = df_filtered["CoverCropGroup"].value_counts().index.tolist()
    unique_groups = df_filtered["CoverCropGroup"].unique()
    palette = dict(zip(unique_groups, sns.color_palette("Set2", len(unique_groups))))

    g = sns.catplot(
        data=df_filtered,
        x="CoverCropGroup",
        y="BackgroundSOC",
        col="Country",
        hue="CoverCropGroup",
        kind="violin",
        col_wrap=3,
        height=4,
        aspect=1.2,
        palette=palette,
        legend=False,
        order=group_order,
        cut=0,
        inner="quartile",
        scale="area"
    )
    g.set_titles("Country: {col_name}")
    g.set_axis_labels("Cover Crop Group", "Soil Organic Carbon (%)")
    g.set_xticklabels(rotation=45, ha="right")
    g.fig.subplots_adjust(top=0.9, right=0.85)
    g.fig.suptitle("SOC Distribution by Cover Crop Group within Each Top Country", fontsize=15)
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=palette[group], label=group) for group in group_order]
    g.fig.legend(
        handles=legend_patches,
        title="Cover Crop Group",
        loc="center left",
        bbox_to_anchor=(0.91, 0.5),
        frameon=True
    )
    sns.despine(trim=True)
    return g.fig

def tukey_soc_plot(df):
    from statsmodels.stats.multicomp import MultiComparison
    df_clean = df.dropna(subset=["CoverCropGroup", "BackgroundSOC"])
    mc = MultiComparison(df_clean["BackgroundSOC"], df_clean["CoverCropGroup"])
    tukey_result = mc.tukeyhsd()
    tukey_summary = pd.DataFrame(
        data=tukey_result._results_table.data[1:],
        columns=tukey_result._results_table.data[0]
    )
    tukey_summary = tukey_summary[tukey_summary["reject"] == True]
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(
        data=tukey_summary,
        x="group1",
        y="meandiff",
        hue="group2",
        dodge=True,
        palette="tab10",
        ax=ax
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Group 1")
    ax.set_ylabel("Mean SOC Difference (%)")
    ax.set_title("Tukey HSD: Significant SOC Differences Between Cover Crop Groups")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="Group 2", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

def heatmap_spearman(df, indicators):
    corr_data = df[indicators].dropna()
    corr_matrix = corr_data.corr(method='spearman')
    
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                linewidths=0.5, square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Spearman Correlation Among Selected Variables', fontsize=14)
    return fig