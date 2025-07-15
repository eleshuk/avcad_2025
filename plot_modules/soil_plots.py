import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="whitegrid")

def boxplot_soil_bd_by_conservation(df):
    top_countries = df['Country'].value_counts().nlargest(5).index
    filtered_df = df[df['Country'].isin(top_countries)]
    valid_types = filtered_df.groupby("Conservation_Type")["SoilBD"].apply(lambda x: x.notna().sum()).loc[lambda x: x > 0].index
    sorted_valid_order = (
        filtered_df[filtered_df["Conservation_Type"].isin(valid_types)]
        .groupby("Conservation_Type")["SoilBD"]
        .median()
        .sort_values()
        .index
    )
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(
        data=filtered_df[filtered_df["Conservation_Type"].isin(valid_types)],
        x="Conservation_Type",
        y="SoilBD",
        hue="Country",
        order=sorted_valid_order,
        palette="Set2",
        ax=ax
    )
    ax.set_title("Soil Bulk Density by Conservation Type (Top 5 Countries)")
    ax.set_ylabel("Soil Bulk Density (g/cm³)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Country", bbox_to_anchor=(1.05, 1), loc="upper left")
    return fig

def barplot_mean_sd_soil_bd(df):
    top_countries = df['Country'].value_counts().nlargest(5).index
    filtered_df = df[df['Country'].isin(top_countries)]
    valid_types = filtered_df.groupby("Conservation_Type")["SoilBD"].apply(lambda x: x.notna().sum()).loc[lambda x: x > 0].index
    filtered_df = filtered_df[filtered_df["Conservation_Type"].isin(valid_types)]
    bar_data = (
        filtered_df.groupby(['Conservation_Type', 'Country'])['SoilBD']
        .agg(['mean', 'std'])
        .reset_index()
    )
    mean_order = (
        bar_data.groupby("Conservation_Type")["mean"]
        .mean()
        .sort_values()
        .index
    )
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(
        data=bar_data,
        x="Conservation_Type",
        y="mean",
        hue="Country",
        order=mean_order,
        palette="Set2",
        errorbar="sd",
        err_kws={'linewidth': 1.5},
        capsize=0.1,
        ax=ax
    )
    ax.set_title("Mean Soil Bulk Density by Conservation Type (± SD)")
    ax.set_ylabel("Mean Soil Bulk Density (g/cm³)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Country", bbox_to_anchor=(1.05, 1), loc="upper left")
    return fig

def heatmap_soil_indicators(df, indicators):
    corr_data = df[indicators].dropna()
    fig, ax = plt.subplots(figsize=(8, 8))  # square shape
    sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', ax=ax, square=True)
    ax.set_title('Correlation Between Soil Health Indicators')
    return fig

import matplotlib.gridspec as gridspec

# def boxplot_multi(df, indicators, top_n=8):
#     n = len(indicators)
#     # fig = plt.figure(figsize=(14, 9 * n))
#     fig = plt.figure(figsize=(14, 9 * n), constrained_layout=True)
#     gs = gridspec.GridSpec(n, 1, height_ratios=[1] * n)
#     axes = []

#     for i, indicator in enumerate(indicators):
#         ax = fig.add_subplot(gs[i])
#         axes.append(ax)
#         temp_df = df[['Conservation_Type', 'Country', indicator]].dropna()
#         top_types = temp_df['Conservation_Type'].value_counts().head(top_n).index
#         temp_df = temp_df[temp_df['Conservation_Type'].isin(top_types)]

#         sns.boxplot(
#             data=temp_df,
#             x='Conservation_Type',
#             y=indicator,
#             hue='Country',
#             palette='Set2',
#             ax=ax,
#             width=1.5
#         )

#         ax.set_title(f'{indicator} by Conservation Practice and Country', fontsize=14)
#         ax.set_xlabel('Conservation Type', fontsize=12)
#         ax.set_ylabel(indicator, fontsize=12)
#         ax.tick_params(axis='x', rotation=45)
#         ax.legend().remove()

#     # Shared legend
#     handles, labels = axes[-1].get_legend_handles_labels()
#     fig.legend(handles, labels, title="Country", bbox_to_anchor=(1.02, 0.5), loc='center left')
#     # fig.subplots_adjust(hspace=0.6)
#     return fig

def boxplot_multi(df, indicators, top_n=8):
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import seaborn as sns

    n = len(indicators)
    fig = plt.figure(figsize=(14, 9 * n))
    # gs = gridspec.GridSpec(n, 1, height_ratios=[1] * n, left=0.15, right=0.75)  # widen left & right margins
    gs = gridspec.GridSpec(n, 1, height_ratios=[1] * n, left=0.1, right=0.7, top=0.95, bottom=0.05, hspace=0.6)

    axes = []

    for i, indicator in enumerate(indicators):
        ax = fig.add_subplot(gs[i])
        axes.append(ax)

        temp_df = df[['Conservation_Type', 'Country', indicator]].dropna()
        top_types = temp_df['Conservation_Type'].value_counts().head(top_n).index
        temp_df = temp_df[temp_df['Conservation_Type'].isin(top_types)]

        sns.boxplot(
            data=temp_df,
            x='Conservation_Type',
            y=indicator,
            hue='Country',
            palette='Set2',
            ax=ax,
            width=0.6  # was 1.5 — made narrower to avoid overflow
        )

        ax.set_title(f'{indicator} by Conservation Practice and Country', fontsize=14)
        ax.set_xlabel('Conservation Type', fontsize=12)
        ax.set_ylabel(indicator, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.legend().remove()

    # Shared legend
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Country", bbox_to_anchor=(0.82, 0.5), loc='center left')
    # fig.legend(handles, labels, title="Country", bbox_to_anchor=(1.01, 0.5), loc="center left")


    # Force tight layout
    # fig.tight_layout(rect=[1, 1, 1, 1])
    return fig


def violinplot_multi(df, indicators, top_n=8):
    n = len(indicators)
    # fig = plt.figure(figsize=(14, 9 * n))
    fig = plt.figure(figsize=(14, 9 * n), constrained_layout=True)
    gs = gridspec.GridSpec(n, 1, height_ratios=[1] * n)
    axes = []

    for i, indicator in enumerate(indicators):
        ax = fig.add_subplot(gs[i])
        axes.append(ax)
        temp_df = df[['Conservation_Type', 'Country', indicator]].dropna()
        top_types = temp_df['Conservation_Type'].value_counts().head(top_n).index
        temp_df = temp_df[temp_df['Conservation_Type'].isin(top_types)]

        sns.violinplot(
            data=temp_df,
            x='Conservation_Type',
            y=indicator,
            hue='Country',
            palette='Set2',
            ax=ax,
            width=1.5
        )

        ax.set_title(f'{indicator} by Conservation Practice and Country', fontsize=14)
        ax.set_xlabel('Conservation Type', fontsize=12)
        ax.set_ylabel(indicator, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.legend().remove()

    # Shared legend
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Country", bbox_to_anchor=(1.02, 0.5), loc='center left')
    # fig.subplots_adjust(hspace=0.6)
    return fig


