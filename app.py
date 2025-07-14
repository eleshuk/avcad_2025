from shiny import App, ui, render
import pandas as pd
import plot_modules.soil_plots as sp
import plot_modules.cover_crop_plots as cp
import plot_modules.soil_crop_yield_plots as scp  # NEW MODULE

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
            ui.output_plot("boxplot_soilbd"),
            ui.br(),
            ui.output_plot("barplot_meansd"),
            ui.br(),
            ui.row(
                ui.column(6, ui.output_plot("corr_heatmap"), align="center", offset=3)
            ),
            ui.br(),
            ui.output_plot("boxplot_multiple", height="1400px"),
            ui.br(),
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
            ui.output_plot("yield_heatmap"),
            ui.br(),
            ui.br(),
            ui.output_plot("yield_by_soil_crop"),
            ui.br(),
            ui.br(),
            ui.output_plot("violin_yield_by_cover_crop"),
            ui.br(),
            ui.br(),
            ui.output_plot("scaled_yield_heatmap"),
            ui.br(),
            ui.br(),
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