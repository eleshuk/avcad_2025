import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols


def prepare_soil_crop_data(df):
    df = df.copy()
    df = df[[
        'SoilFamily', 'GrainCropGroup', 'CoverCropGroup', 'Yield_C', 'Yield_T'
    ]].dropna()

    df_t = df.rename(columns={'Yield_T': 'Yield'}).copy()
    df_t['Treatment'] = '_T'

    df_c = df.rename(columns={'Yield_C': 'Yield'}).copy()
    df_c['Treatment'] = '_C'
    df_c['CoverCropGroup'] = 'None'

    df_combined = pd.concat([df_t, df_c], ignore_index=True)

    df_combined.drop(columns=['Yield_C', 'Yield_T'], errors='ignore', inplace=True)

    df_combined['GrainCropGroup'] = df_combined['GrainCropGroup'].replace({
        'CS': 'Corn-soybean', 'CSO': 'Corn-soybean',
        'CW': 'Corn-wheat', 'CO': 'Corn-oat',
        'WO': 'Wheat-oat', 'CWO': 'Corn-wheat-millet',
        'CSW': 'Corn-soybean-wheat',
        'AVG': 'Unknown', 'Other': 'Unknown',
        'CV': 'Vegetable', 'CVO': 'Vegetable', 'WV': 'Vegetable'
    })
    df_combined = df_combined[df_combined['GrainCropGroup'] != 'MTT']

    scaler = MinMaxScaler()
    df_combined['Yield_scaled'] = scaler.fit_transform(df_combined[['Yield']])
    return df_combined


def plot_soil_crop_heatmap(df):
    df = df.dropna(subset=['SoilFamily', 'GrainCropGroup'])
    top_soils = df['SoilFamily'].value_counts().nlargest(8).index
    df_top = df[df['SoilFamily'].isin(top_soils)]

    pivot = df_top.pivot_table(index='SoilFamily', columns='GrainCropGroup', aggfunc='size', fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='d', cmap='viridis', ax=ax)
    ax.set_title('Soil Family vs. Grain Crop Group (Top 8)')
    ax.set_xlabel('Grain Crop Group')
    ax.set_ylabel('Soil Family')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_yield_by_soil_crop(df):
    df.columns = df.columns.str.lower()
    df = df.dropna(subset=['soilfamily', 'graincropgroup', 'yield'])
    df_grouped = df.groupby(['soilfamily', 'graincropgroup'])['yield'].mean().reset_index()

    order = df_grouped.groupby('soilfamily')['yield'].mean().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(30, 12)) 
    sns.barplot(
        data=df_grouped,
        x='soilfamily',
        y='yield',
        hue='graincropgroup',
        order=order,
        ax=ax,
        palette='Dark2',
        # edgecolor='black',
        dodge = False,
        linewidth=1
    )

    ax.set_title('Average Yield by Soil Family and Crop Group', fontsize=16)
    ax.set_xlabel('Soil Family', fontsize=12)
    ax.set_ylabel('Average Yield', fontsize=12)
    ax.legend(title='Grain Crop Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=8)
    fig.subplots_adjust(bottom=0.35, top=0.9)
    plt.tight_layout()
    return fig


def violin_yield_by_cover_crop(df):
    df = df.dropna(subset=['SoilFamily', 'CoverCropGroup', 'Yield'])

    df['CoverCropGroup'] = df['CoverCropGroup'].replace({
        'LL': 'Legume', 'Legume_Tree': 'Legume',
        'LG': 'Mixed', 'BG': 'Mixed', 'AVG': 'Mixed', 'MOT': 'Mixed',
        'Not_available': 'Unknown', 'No': 'No', 'None': 'No',
        'BroadleafTree': 'Broadleaf'
    })

    df['Scaled_Yield'] = df.groupby('SoilFamily')['Yield'].transform(
        lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).flatten()
    )

    fig, ax = plt.subplots(figsize=(16, 9))
    sns.violinplot(data=df, x='SoilFamily', y='Scaled_Yield', hue='CoverCropGroup', split=True, inner='quart', ax=ax)
    ax.set_title('Yield by Soil Family and Cover Crop Group')
    ax.set_xlabel('Soil Family')
    ax.set_ylabel('Scaled Yield')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def heatmap_scaled_yield(df):
    if 'Scaled_Yield' not in df.columns:
        df['Scaled_Yield'] = df.groupby('SoilFamily')['Yield'].transform(
            lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).flatten()
        )

    pivot = df.pivot_table(index='SoilFamily', columns='CoverCropGroup', values='Scaled_Yield', aggfunc='mean')

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Average Scaled Yield by Soil Family and Cover Crop Group')
    ax.set_xlabel('Cover Crop Group')
    ax.set_ylabel('Soil Family')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_anova_cover_crop_effect(df):
    # Use lowercase column names to match your df
    df_anova = df[['covercropgroup', 'yield_scaled']].dropna().copy()

    try:
        model = ols('yield_scaled ~ C(covercropgroup)', data=df_anova).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df_anova, x='covercropgroup', y='yield_scaled', palette='viridis', ax=ax)
        ax.set_title('Scaled Yield by Cover Crop Group')
        ax.set_xlabel('Cover Crop Group')
        ax.set_ylabel('Scaled Yield')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        return fig

    except Exception as e:
        print("ANOVA error:", e)
        return None

