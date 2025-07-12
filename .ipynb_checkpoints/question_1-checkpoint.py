#!/usr/bin/env python
# coding: utf-8

# In[82]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings("ignore", message="covariance of constraints does not have full rank")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")


# ## Import data

# In[25]:


df = pd.read_excel('SoilHealthDB_V2.xlsx')
# df.head()


# In[65]:


pd.set_option('display.max_columns', None)
# Corrected management columns (replacing the incorrect one)
management_columns_corrected = [
    'CoverCrop', 'CoverCropGroup', 'TimeAfterCoverCrop', 'CC_termination_date',
    'Rotation_C', 'Rotation_T',
    'Tillage_C', 'Tillage_T',
    'Fertilization_C',
    'Conservation_Type', 'Conservation_Discription'
]

# Identifier and location columns
identifier_columns = ['StudyID', 'ExperimentID', 'Country', 'SiteInfor']

# Soil health indicator column prefixes
soil_health_prefixes = [
    'SoilBD', 'SoilpH', 'OC_', 'N_', 'P_', 'K_', 'CEC_', 'EC_', 'BS_',
    'Porosity_', 'Penetration_', 'Infiltration_', 'Erosion_', 'Runoff_', 'Leaching_',
    'ST_', 'SWC_', 'AWHC_',
    'Weed_', 'Diseases_', 'Pests_', 'SoilFauna_', 'Fungal_', 'OtherMicrobial_',
    'Enzyme_', 'Cmina_', 'Nmina_', 'NxO_', 'SIR_', 'CO2BTest_', 'CO2_', 'CH4_',
    'MBC_', 'MBN_', 'SQI', 'ESS'
]

# Select soil health columns based on prefixes
soil_health_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in soil_health_prefixes)]

# Combine all columns
final_columns = identifier_columns + management_columns_corrected + soil_health_columns

# Filter and display the DataFrame
full_selection_df = df[final_columns]

full_selection_df.head()


# ### Drop NaN rows

# In[40]:


# Choose the column after which to check for NaNs (e.g., after 'A')
col_index = full_selection_df.columns.get_loc('Conservation_Discription') + 1  # Get index of next column after 'A'
subset = full_selection_df.columns[col_index:]

# Drop rows where all columns after 'A' are NaN
df_filtered = full_selection_df[~full_selection_df[subset].isna().all(axis=1)]
df_filtered.head()


# ## Summary Statistics

# In[41]:


# Select a small set of key soil health indicators for demonstration
soil_indicators = ['SoilpH', 'OC_C', 'SoilBD', 'MBC_C', 'Porosity_C']
group_vars = ['Conservation_Type', 'Country']

# 1. Grouped Summary Table
summary_stats = df_filtered.groupby(group_vars)[soil_indicators].agg(['mean', 'std']).reset_index()
summary_stats.head()


# ## Exploratory data analysis

# ### Boxplots - Soil Organic Carbon by Conservation Type and Country

# In[53]:


# Redefine the filtered_df based on top 5 countries again since it's missing
top_countries = (
    full_selection_df['Country']
    .value_counts()
    .nlargest(5)
    .index
)

filtered_df = full_selection_df[full_selection_df['Country'].isin(top_countries)]

# Identify which Conservation_Types have data
valid_types = (
    filtered_df.groupby("Conservation_Type")["SoilBD"]
    .apply(lambda x: x.notna().sum())
    .loc[lambda x: x > 0]
    .index
)

# Sort remaining types by median SoilBD
sorted_valid_order = (
    filtered_df[filtered_df["Conservation_Type"].isin(valid_types)]
    .groupby("Conservation_Type")["SoilBD"]
    .median()
    .sort_values()
    .index
)

# Plot cleaned version
plt.figure(figsize=(14, 7))
sns.set_theme(style="whitegrid")
sns.boxplot(
    data=filtered_df[filtered_df["Conservation_Type"].isin(valid_types)],
    x="Conservation_Type",
    y="SoilBD",
    hue="Country",
    order=sorted_valid_order,
    palette="Set2"
)

# Formatting
plt.title("Soil Bulk Density by Conservation Type (Top 5 Countries)", fontsize=14, weight='bold')
plt.ylabel("Soil Bulk Density (g/cm³)")
plt.xlabel("")
plt.xticks(rotation=45)
plt.legend(title="Country", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


# ### Mean and SD barplot

# In[56]:


import seaborn as sns
import matplotlib.pyplot as plt

# Filter to top 5 countries with most data
top_countries = (
    full_selection_df['Country']
    .value_counts()
    .nlargest(5)
    .index
)

filtered_df = full_selection_df[full_selection_df['Country'].isin(top_countries)]

# Remove unused Conservation Types
valid_types = (
    filtered_df.groupby("Conservation_Type")["SoilBD"]
    .apply(lambda x: x.notna().sum())
    .loc[lambda x: x > 0]
    .index
)

filtered_df = filtered_df[filtered_df["Conservation_Type"].isin(valid_types)]

# Compute mean and standard deviation
bar_data = (
    filtered_df.groupby(['Conservation_Type', 'Country'])['SoilBD']
    .agg(['mean', 'std'])
    .reset_index()
)

# Sort conservation types by average mean SoilBD
mean_order = (
    bar_data.groupby("Conservation_Type")["mean"]
    .mean()
    .sort_values()
    .index
)

# Plot using updated seaborn API
plt.figure(figsize=(14, 7))
sns.set_theme(style="whitegrid")
sns.barplot(
    data=bar_data,
    x="Conservation_Type",
    y="mean",
    hue="Country",
    order=mean_order,
    palette="Set2",
    errorbar=("sd"),
    err_kws={'linewidth': 1.5},
    capsize=0.1
)

# Format plot
plt.title("Mean Soil Bulk Density by Conservation Type (± SD)", fontsize=14, weight='bold')
plt.ylabel("Mean Soil Bulk Density (g/cm³)")
plt.xlabel("")
plt.xticks(rotation=45)
plt.legend(title="Country", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


# ### Correlation Heatmap

# In[19]:


corr_data = full_selection_df[soil_indicators].dropna()
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', ax=ax3)
ax3.set_title('Correlation Between Soil Health Indicators')


# ## Inferential statistics

# ### Two way ANOVA

# In[86]:


# Define list of numeric soil indicators
numeric_indicators = [
    col for col in soil_health_columns
    if pd.api.types.is_numeric_dtype(df_filtered[col]) and df_filtered[col].notna().sum() >= 30
]

# Run Two-Way ANOVA for each indicator
anova_results = []

for indicator in numeric_indicators:
    temp_df = df_filtered[['Conservation_Type', 'Country', indicator]].dropna()
    if temp_df['Conservation_Type'].nunique() < 2 or temp_df['Country'].nunique() < 2:
        continue
    try:
        formula = f'{indicator} ~ C(Conservation_Type) + C(Country) + C(Conservation_Type):C(Country)'
        model = ols(formula, data=temp_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        anova_results.append({
            'Soil_Indicator': indicator,
            'p_Conservation': anova_table.loc['C(Conservation_Type)', 'PR(>F)'],
            'p_Country': anova_table.loc['C(Country)', 'PR(>F)'],
            'p_Interaction': anova_table.loc['C(Conservation_Type):C(Country)', 'PR(>F)']
        })
    except Exception as e:
        print(f"Skipped {indicator}: {e}")

# Format the ANOVA summary
anova_df = pd.DataFrame(anova_results).sort_values(by='p_Conservation')
anova_df = anova_df.round(4)
significant_indicators = anova_df[anova_df['p_Conservation'] < 0.05]['Soil_Indicator'].tolist()

# Tukey HSD tests for significant indicators
tukey_all = []

for indicator in significant_indicators:
    temp_df = df_filtered[['Conservation_Type', indicator]].dropna()
    try:
        tukey = pairwise_tukeyhsd(endog=temp_df[indicator], groups=temp_df['Conservation_Type'], alpha=0.05)
        tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
        tukey_df = tukey_df[tukey_df['reject'] == True]
        tukey_df['Indicator'] = indicator
        tukey_all.append(tukey_df)
    except Exception as e:
        print(f"Tukey failed for {indicator}: {e}")

# Combine Tukey results into final table
if tukey_all:
    tukey_results_df = pd.concat(tukey_all, ignore_index=True)
    tukey_results_df['meandiff'] = tukey_results_df['meandiff'].astype(float).round(3)
    tukey_results_df['p-adj'] = tukey_results_df['p-adj'].astype(float).round(4)
    display(tukey_results_df[['Indicator', 'group1', 'group2', 'meandiff', 'p-adj', 'reject']])
else:
    print("No significant Tukey comparisons found.")


# ### Boxplots for OC_C, MBC_C, and SoilpH

# In[100]:


sns.set(style="whitegrid")

indicators = ['OC_C', 'MBC_C', 'SoilpH']
top_n = 8  # most frequent conservation types

fig, axes = plt.subplots(len(indicators), 1, figsize=(14, 18), sharex=False)

for i, indicator in enumerate(indicators):
    temp_df = df_filtered[['Conservation_Type', 'Country', indicator]].dropna()
    top_types = temp_df['Conservation_Type'].value_counts().head(top_n).index
    temp_df = temp_df[temp_df['Conservation_Type'].isin(top_types)]

    sns.boxplot(
        data=temp_df,
        x='Conservation_Type',
        y=indicator,
        hue='Country',
        palette='Set2',
        ax=axes[i],
        width=0.9
    )

    axes[i].set_title(f'{indicator} by Conservation Practice and Country', fontsize=14)
    axes[i].set_xlabel('Conservation Type', fontsize=12)
    axes[i].set_ylabel(indicator, fontsize=12)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].legend().remove()

# Shared legend outside plot
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, title="Country", bbox_to_anchor=(1.02, 0.5), loc='center left')
plt.tight_layout()
plt.show()


# ### Same, but violin plots

# In[97]:


fig, axes = plt.subplots(len(indicators), 1, figsize=(14, 18), sharex=False)

for i, indicator in enumerate(indicators):
    temp_df = df_filtered[['Conservation_Type', 'Country', indicator]].dropna()
    top_types = temp_df['Conservation_Type'].value_counts().head(top_n).index
    temp_df = temp_df[temp_df['Conservation_Type'].isin(top_types)]

    sns.violinplot(
        data=temp_df,
        x='Conservation_Type',
        y=indicator,
        hue='Country',
        palette='Set2',
        ax=axes[i],
        cut=0,
        split=False
    )

    axes[i].set_title(f'{indicator} by Conservation Practice and Country (Violin)', fontsize=14)
    axes[i].set_xlabel('Conservation Type', fontsize=12)
    axes[i].set_ylabel(indicator, fontsize=12)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].legend().remove()

# Shared legend
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, title="Country", bbox_to_anchor=(1.02, 0.5), loc='center left')
plt.tight_layout()
plt.show()


# In[90]:


# Make it pretty
# Create a clean summary table
summary_table = tukey_results_df[['Indicator', 'group1', 'group2', 'meandiff', 'p-adj', 'reject']].copy()
summary_table.columns = ['Indicator', 'Group 1', 'Group 2', 'Mean Diff', 'Adjusted p', 'Significant']
summary_table['Significant'] = summary_table['Significant'].map({True: '✅', False: '❌'})
summary_table = summary_table.sort_values(by=['Indicator', 'Adjusted p'])

# Display top 10
summary_table.tail(10)


# In[ ]:




