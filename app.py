from shiny import App, ui, render
import pandas as pd
import plot_modules.soil_plots as sp
import plot_modules.cover_crop_plots as cp
import plot_modules.soil_crop_yield_plots as scp 

# Load data
df = pd.read_excel("data/SoilHealthDB_V2.xlsx")

# Soil indicators tab
prefixes = ['SoilBD', 'SoilpH', 'OC_', 'MBC_', 'Porosity_']
soil_cols = [col for col in df.columns if any(col.startswith(p) for p in prefixes)]
df1 = df.copy()
df_filtered = df1[~df1[soil_cols].isna().all(axis=1)]

# Cover crop tab
df_soc = df[['CoverCropGroup', 'BackgroundSOC', 'Country']].dropna()
df_soc['Country'] = df_soc['Country'].astype(str).str.strip().str.title()

# Soil/crop yield tab
df_yield = scp.prepare_soil_crop_data(df.copy())
df_soil_multiple_crops = df_yield[df_yield['SoilFamily'].isin(
    df_yield.groupby('SoilFamily')['GrainCropGroup'].nunique().loc[lambda x: x >= 3].index
)].copy()

app_ui = ui.page_fluid(
    ui.navset_tab(
        ui.nav_panel("Conservation Plots",
            ui.br(),
            ui.h3("Exploratory Analysis"),
            ui.p(f"""
                This plot shows a comparison of soil physical conditions
                under different management strategies across countries. The results suggest that while most
                conservation types fall within a similar bulk density range, some
                practices, particularly combinations involving no-till (NT) and cover crops (CC), tend to exhibit
                lower median SoilBD in several countries.
                 """),
            ui.output_plot("boxplot_soilbd"),
            ui.br(),
            ui.br(),
            ui.p(f"""
                    Barplot showing the mean SoilBD ± standard deviation for each conservation
                    type across the top five countries. The data was grouped by Conservation_Type and Country to
                    calculate the average and variability in SoilBD
                 """),
            ui.output_plot("barplot_meansd"),
            ui.br(),
            ui.br(),
            ui.h3("Inferrential Statistics"),
            ui.p(f"""
                Correlation matrix of key soil indicators, demonstrates strong
                positive correlations between biological indicators like microbial biomass (MBC_C) and organic
                carbon (OC_C), suggesting that improvements in organic matter content may enhance microbial
                activity.
                """),
            ui.row(
                ui.column(6, ui.output_plot("corr_heatmap"), align="center", offset=3)
            ),
            ui.br(),
            ui.br(),
            ui.p(f"""
                Three indicators, OC_C, MBC_C, and SoilpH, showed statistically significant main effects of
                Conservation_Type (p < 0.05) and were selected for deeper investigation. 
                """),
            ui.p(f"""
                Comparative boxplots grouped by conservation type and colored by country, illustrating both
                within-group variability and geographic trends. 
                """),
            ui.output_plot("boxplot_multiple", height="1400px"),
            ui.br(),
            ui.br(),
            ui.p(f"""
                These visualizations suggest that conservation
                practices can meaningfully influence soil conditions, although the magnitude and direction of
                effects vary across countries, consistent with the presence of significant interaction terms in the
                ANOVA results.
                """),
            ui.br(),
            ui.br(),
            ui.p(
                ui.tags.b(f"""
                 Violin plots showing the same visualization as the boxplots.
                """)),
            ui.output_plot("violin_multiple", height="1400px")
        ),
        ui.nav_panel("Cover Crop Plots",
            ui.row(
                ui.column(6, ui.output_plot("heatmap_spearman"), align="center", offset=3)
            ),
            ui.output_plot("boxplot_soc"),
            ui.output_plot("barplot_country_freq"),
            ui.row(
                ui.column(6, ui.output_plot("soc_heatmap"), align="center", offset=3)
            ),
            ui.output_plot("violin_soc_by_country", height="900px"),
            ui.output_plot("tukey_soc_diff", height="700px")
        ),
        ui.nav_panel("Soil & Crop Yield",
            ui.br(),
            ui.br(),
            ui.p("This plot shows the average yield of each crop for each soil family. It can be seen that the top yielding soils are growing vegetables, with grain crops being more common in the less productive soils."),
            ui.output_plot("yield_by_soil_crop"),
            ui.br(),
            ui.br(),
            ui.p("This plot shows the amount of records of each crop for the top 8 soil families with the highest amount of records."),
            ui.output_plot("yield_heatmap"),
            ui.br(),
            ui.br(),
            ui.p("This plot shows the yield per soil family (soil families with 3 or more crop groups), for no cover crop and with rye, mixed, or brassica cover crops. There does not appear to be a significant difference in yield between the cover crop groups and no cover crop, however comparisns are very restricted, so no broad conclusion can be reached about cover crops and crop growth."),
            ui.output_plot("violin_yield_by_cover_crop"),
            ui.br(),
            ui.br(),
            ui.p("This plot shows the correlated between yield and cover crop (or none) in the five soil groups with atleast 3 crops growing in them. As can be seen, correlations are very similar with or without cover crops."),
            ui.output_plot("scaled_yield_heatmap"),
            ui.br(),
            ui.br(),
            ui.p("This plot shows the scaled yield for each cover crop, with most cover crops seeming to have little impact on the yield, with the exception of grass and BG (mix of brassica and grass)."),
            ui.output_plot("anova_boxplot")
        )
    )
)

def server(input, output, session):
    # Tab 1 — Soil indicators
    @output
    @render.plot
    def boxplot_soilbd():
        return sp.boxplot_soil_bd_by_conservation(df1)

    @output
    @render.plot
    def barplot_meansd():
        return sp.barplot_mean_sd_soil_bd(df1)

    @output
    @render.plot
    def corr_heatmap():
        return sp.heatmap_soil_indicators(df_filtered, ['SoilpH', 'OC_C', 'SoilBD', 'MBC_C', 'Porosity_C'])

    @output
    @render.plot
    def boxplot_multiple():
        return sp.boxplot_multi(df_filtered, ['OC_C', 'MBC_C', 'SoilpH'])

    @output
    @render.plot
    def violin_multiple():
        return sp.violinplot_multi(df_filtered, ['OC_C', 'MBC_C', 'SoilpH'])

    # Tab 2 — Cover crop plots
    @output
    @render.plot
    def heatmap_spearman():
        return cp.heatmap_spearman(df, ["BackgroundSOC", "Duration", "SoilpH"])

    @output
    @render.plot
    def boxplot_soc():
        return cp.boxplot_soc(df_soc)

    @output
    @render.plot
    def barplot_country_freq():
        return cp.barplot_country_freq(df_soc)

    @output
    @render.plot
    def soc_heatmap():
        return cp.heatmap_soc_means(df_soc)

    @output
    @render.plot
    def violin_soc_by_country():
        return cp.violinplot_soc_by_country(df_soc)

    @output
    @render.plot
    def tukey_soc_diff():
        return cp.tukey_soc_plot(df_soc)

    # Tab 3 — Soil & Crop Yield
    @output
    @render.plot
    def yield_heatmap():
        return scp.plot_soil_crop_heatmap(df_yield)

    @output
    @render.plot
    def yield_by_soil_crop():
        return scp.plot_yield_by_soil_crop(df_yield)

    @output
    @render.plot
    def violin_yield_by_cover_crop():
        return scp.violin_yield_by_cover_crop(df_soil_multiple_crops)

    @output
    @render.plot
    def scaled_yield_heatmap():
        return scp.heatmap_scaled_yield(df_soil_multiple_crops)

    @output
    @render.plot
    def anova_boxplot():
        return scp.plot_anova_cover_crop_effect(df_yield)

app = App(app_ui, server)

if __name__ == "__main__":
    app.run()