from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


REPO_DIR = Path(__file__).resolve().parents[1]

if str(REPO_DIR) not in sys.path:
    sys.path.append(str(REPO_DIR))

import reeds  # noqa: E402


CARBON_CMAP = "YlOrRd"
WATER_CMAP = "YlGnBu"
LITERS_PER_GALLON = 3.785411784
KWH_PER_MWH = 1000.0
GALLONS_PER_MGAL = 1_000_000.0
AVG_START_YEAR = 2026


LOG10_TICK_FORMATTER = FuncFormatter(lambda x, pos: rf"$10^{{{x:.0f}}}$")


def calc_factor_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["grid_carbon_factor_ton_per_mwh"] = df["co2e_metric_tons"] / df["generation_mwh"]
    df["grid_carbon_factor_kgco2eq_per_kwh"] = (
        df["grid_carbon_factor_ton_per_mwh"] * 1000.0 / KWH_PER_MWH
    )
    df["grid_thermoelectric_water_factor_mgal_per_mwh"] = (
        df["thermoelectric_water_consumption_mgal"] / df["generation_mwh"]
    )
    df["grid_thermoelectric_water_factor_l_per_kwh"] = (
        df["grid_thermoelectric_water_factor_mgal_per_mwh"]
        * GALLONS_PER_MGAL
        * LITERS_PER_GALLON
        / KWH_PER_MWH
    )
    df["grid_water_factor_mgal_per_mwh"] = df["total_water_consumption_mgal"] / df["generation_mwh"]
    df["grid_water_factor_l_per_kwh"] = (
        df["grid_water_factor_mgal_per_mwh"]
        * GALLONS_PER_MGAL
        * LITERS_PER_GALLON
        / KWH_PER_MWH
    )

    invalid_generation = df["generation_mwh"] <= 0
    factor_cols = [
        "grid_carbon_factor_ton_per_mwh",
        "grid_carbon_factor_kgco2eq_per_kwh",
        "grid_thermoelectric_water_factor_mgal_per_mwh",
        "grid_thermoelectric_water_factor_l_per_kwh",
        "grid_water_factor_mgal_per_mwh",
        "grid_water_factor_l_per_kwh",
    ]
    df.loc[invalid_generation, factor_cols] = np.nan
    inf_mask = ~np.isfinite(df[factor_cols])
    df.loc[inf_mask.any(axis=1), factor_cols] = np.nan
    return df


def load_hydro_evaporation_by_region(case_dir: Path, hierarchy: pd.DataFrame) -> pd.DataFrame:
    outputs_dir = case_dir / "outputs"
    evap_path = Path(__file__).resolve().parent / "water_evaporation.csv"

    hydro_gen = pd.read_csv(outputs_dir / "gen_ann.csv")
    hydro_gen = (
        hydro_gen.loc[hydro_gen["i"].str.startswith("hyd", na=False)]
        .groupby(["r", "t"], as_index=False)["Value"]
        .sum()
        .rename(columns={"Value": "hydro_generation_mwh"})
    )

    evap_factors = pd.read_csv(evap_path, usecols=["st", "L/kWh"])
    evap_factors = evap_factors.rename(columns={"L/kWh": "hydro_evaporation_l_per_kwh"})

    hydro_evap = hydro_gen.merge(
        hierarchy.reset_index()[["r", "st"]],
        on="r",
        how="left",
    ).merge(
        evap_factors,
        on="st",
        how="left",
    )
    hydro_evap["hydro_evaporation_liters"] = (
        hydro_evap["hydro_generation_mwh"]
        * KWH_PER_MWH
        * hydro_evap["hydro_evaporation_l_per_kwh"].fillna(0.0)
    )
    hydro_evap["hydro_evaporation_mgal"] = (
        hydro_evap["hydro_evaporation_liters"] / LITERS_PER_GALLON / GALLONS_PER_MGAL
    )
    return hydro_evap[["r", "t", "hydro_generation_mwh", "hydro_evaporation_mgal"]]


def load_ba_factor_data(case_dir: Path) -> pd.DataFrame:
    outputs_dir = case_dir / "outputs"
    hierarchy = reeds.io.get_hierarchy(str(case_dir))

    emit = pd.read_csv(outputs_dir / "emit_r.csv")
    emit = (
        emit.loc[emit["etype"].eq("process") & emit["eall"].eq("CO2e")]
        .groupby(["r", "t"], as_index=False)["Value"]
        .sum()
        .rename(columns={"Value": "co2e_metric_tons"})
    )

    gen = pd.read_csv(outputs_dir / "gen_ann.csv")
    gen = (
        gen.groupby(["r", "t"], as_index=False)["Value"]
        .sum()
        .rename(columns={"Value": "generation_mwh"})
    )

    water = pd.read_csv(outputs_dir / "water_consumption_ivrt.csv")
    water = (
        water.groupby(["r", "t"], as_index=False)["Value"]
        .sum()
        .rename(columns={"Value": "thermoelectric_water_consumption_mgal"})
    )

    hydro_evap = load_hydro_evaporation_by_region(case_dir, hierarchy)

    factors = (
        gen.merge(emit, on=["r", "t"], how="left")
        .merge(water, on=["r", "t"], how="left")
        .merge(hydro_evap, on=["r", "t"], how="left")
        .fillna(
            {
                "co2e_metric_tons": 0.0,
                "thermoelectric_water_consumption_mgal": 0.0,
                "hydro_generation_mwh": 0.0,
                "hydro_evaporation_mgal": 0.0,
            }
        )
    )
    factors["total_water_consumption_mgal"] = (
        factors["thermoelectric_water_consumption_mgal"] + factors["hydro_evaporation_mgal"]
    )
    factors = calc_factor_columns(factors)

    factors = factors.merge(hierarchy.reset_index(), on="r", how="left")

    output_columns = [
        "r",
        "t",
        "st",
        "country",
        "generation_mwh",
        "co2e_metric_tons",
        "hydro_generation_mwh",
        "hydro_evaporation_mgal",
        "thermoelectric_water_consumption_mgal",
        "total_water_consumption_mgal",
        "grid_carbon_factor_ton_per_mwh",
        "grid_carbon_factor_kgco2eq_per_kwh",
        "grid_thermoelectric_water_factor_mgal_per_mwh",
        "grid_thermoelectric_water_factor_l_per_kwh",
        "grid_water_factor_mgal_per_mwh",
        "grid_water_factor_l_per_kwh",
    ]
    return factors[output_columns].sort_values(["t", "r"]).reset_index(drop=True)


def aggregate_to_state(ba_factors: pd.DataFrame) -> pd.DataFrame:
    state_region_members = (
        ba_factors[["st", "r"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["st", "r"])
        .groupby("st", as_index=False)["r"]
        .agg(lambda x: "|".join(x))
        .rename(columns={"r": "aggregated_r_list"})
    )

    meta_cols = ["st", "country"]
    meta = (
        ba_factors[meta_cols]
        .dropna(subset=["st"])
        .drop_duplicates()
        .groupby("st", as_index=False)
        .agg(
            country=("country", lambda x: "|".join(sorted(set(x.dropna())))),
        )
    )

    state = (
        ba_factors.groupby(["st", "t"], as_index=False)[
            [
                "generation_mwh",
                "co2e_metric_tons",
                "hydro_generation_mwh",
                "hydro_evaporation_mgal",
                "thermoelectric_water_consumption_mgal",
                "total_water_consumption_mgal",
            ]
        ]
        .sum()
    )
    state = calc_factor_columns(state)
    state = state.merge(meta, on="st", how="left").merge(state_region_members, on="st", how="left")

    output_columns = [
        "st",
        "t",
        "aggregated_r_list",
        "country",
        "generation_mwh",
        "co2e_metric_tons",
        "hydro_generation_mwh",
        "hydro_evaporation_mgal",
        "thermoelectric_water_consumption_mgal",
        "total_water_consumption_mgal",
        "grid_carbon_factor_ton_per_mwh",
        "grid_carbon_factor_kgco2eq_per_kwh",
        "grid_thermoelectric_water_factor_mgal_per_mwh",
        "grid_thermoelectric_water_factor_l_per_kwh",
        "grid_water_factor_mgal_per_mwh",
        "grid_water_factor_l_per_kwh",
    ]
    return state[output_columns].sort_values(["t", "st"]).reset_index(drop=True)


def average_factors_from_year(factors: pd.DataFrame, id_col: str, start_year: int) -> pd.DataFrame:
    avg = factors.loc[factors["t"].ge(start_year)].copy()
    if avg.empty:
        raise ValueError(f"No factor data found for years >= {start_year}")

    group_cols = [id_col]
    if "country" in avg.columns:
        group_cols.append("country")
    if "aggregated_r_list" in avg.columns:
        group_cols.append("aggregated_r_list")

    avg = (
        avg.groupby(group_cols, as_index=False)[
            [
                "generation_mwh",
                "co2e_metric_tons",
                "hydro_generation_mwh",
                "hydro_evaporation_mgal",
                "thermoelectric_water_consumption_mgal",
                "total_water_consumption_mgal",
            ]
        ]
        .sum()
    )
    avg = calc_factor_columns(avg)
    avg["year_range"] = f"{start_year}+"

    output_columns = [id_col, "year_range"]
    if "aggregated_r_list" in avg.columns:
        output_columns.append("aggregated_r_list")
    if "country" in avg.columns:
        output_columns.append("country")
    output_columns += [
        "generation_mwh",
        "co2e_metric_tons",
        "hydro_generation_mwh",
        "hydro_evaporation_mgal",
        "thermoelectric_water_consumption_mgal",
        "total_water_consumption_mgal",
        "grid_carbon_factor_ton_per_mwh",
        "grid_carbon_factor_kgco2eq_per_kwh",
        "grid_thermoelectric_water_factor_mgal_per_mwh",
        "grid_thermoelectric_water_factor_l_per_kwh",
        "grid_water_factor_mgal_per_mwh",
        "grid_water_factor_l_per_kwh",
    ]
    return avg[output_columns].sort_values(id_col).reset_index(drop=True)


def write_factor_outputs(factors: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    factors.to_csv(output_dir / f"{prefix}_carbon_water_factors_by_year.csv", index=False)


def write_average_factor_outputs(
    factors: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    start_year: int,
) -> None:
    factors.to_csv(output_dir / f"{prefix}_carbon_water_factors_avg_{start_year}_onward.csv", index=False)


def get_map(case_dir: Path, level: str) -> gpd.GeoDataFrame:
    dfmap = reeds.io.get_dfmap(str(case_dir), levels=[level, "country"])
    return dfmap[level].copy()


def add_panel(
    ax: plt.Axes,
    geodf: gpd.GeoDataFrame,
    values: pd.DataFrame,
    id_col: str,
    year: int,
    value_column: str,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    merged = geodf.join(values.set_index(id_col)[value_column], how="left")
    merged.plot(
        ax=ax,
        column=value_column,
        cmap=cmap,
        linewidth=0.25,
        edgecolor="#666666",
        vmin=vmin,
        vmax=vmax,
        missing_kwds={"color": "#f2f2f2", "edgecolor": "#999999", "label": "No data"},
    )
    geodf.boundary.plot(ax=ax, linewidth=0.15, color="#777777")
    ax.set_title(f"{title} {year}", fontsize=11)
    ax.axis("off")


def transform_log10_positive(values: pd.Series) -> pd.Series:
    transformed = values.where(values > 0)
    return np.log10(transformed)


def plot_combined_factor_grid(
    factors: pd.DataFrame,
    geodf: gpd.GeoDataFrame,
    output_path: Path,
    id_col: str,
    level_label: str,
) -> None:
    years = sorted(factors["t"].unique())
    carbon_vmax = float(factors["grid_carbon_factor_kgco2eq_per_kwh"].dropna().quantile(0.98))
    thermoelectric_water_vmax = float(
        factors["grid_thermoelectric_water_factor_l_per_kwh"].dropna().quantile(0.98)
    )
    total_water_log_values = transform_log10_positive(factors["grid_water_factor_l_per_kwh"])
    water_vmin = float(total_water_log_values.quantile(0.02))
    water_vmax = float(total_water_log_values.quantile(0.98))

    fig, axes = plt.subplots(len(years), 3, figsize=(18, 4.6 * len(years)), constrained_layout=True)
    if len(years) == 1:
        axes = np.array([axes])

    for row, year in enumerate(years):
        year_values = factors.loc[factors["t"].eq(year)].copy()

        add_panel(
            ax=axes[row, 0],
            geodf=geodf,
            values=year_values[[id_col, "grid_carbon_factor_kgco2eq_per_kwh"]],
            id_col=id_col,
            year=year,
            value_column="grid_carbon_factor_kgco2eq_per_kwh",
            title=f"{level_label} grid carbon factor",
            cmap=CARBON_CMAP,
            vmin=0.0,
            vmax=carbon_vmax,
        )
        add_panel(
            ax=axes[row, 1],
            geodf=geodf,
            values=year_values[[id_col, "grid_thermoelectric_water_factor_l_per_kwh"]],
            id_col=id_col,
            year=year,
            value_column="grid_thermoelectric_water_factor_l_per_kwh",
            title=f"{level_label} grid thermoelectric water factor",
            cmap=WATER_CMAP,
            vmin=0.0,
            vmax=thermoelectric_water_vmax,
        )
        add_panel(
            ax=axes[row, 2],
            geodf=geodf,
            values=year_values[[id_col, "grid_water_factor_l_per_kwh"]].assign(
                grid_water_factor_log10_l_per_kwh=transform_log10_positive(
                    year_values["grid_water_factor_l_per_kwh"]
                )
            )[[id_col, "grid_water_factor_log10_l_per_kwh"]],
            id_col=id_col,
            year=year,
            value_column="grid_water_factor_log10_l_per_kwh",
            title=f"{level_label} grid water factor",
            cmap=WATER_CMAP,
            vmin=water_vmin,
            vmax=water_vmax,
        )

    carbon_sm = plt.cm.ScalarMappable(cmap=CARBON_CMAP, norm=plt.Normalize(vmin=0.0, vmax=carbon_vmax))
    carbon_sm._A = []
    thermoelectric_water_sm = plt.cm.ScalarMappable(
        cmap=WATER_CMAP, norm=plt.Normalize(vmin=0.0, vmax=thermoelectric_water_vmax)
    )
    thermoelectric_water_sm._A = []
    water_sm = plt.cm.ScalarMappable(cmap=WATER_CMAP, norm=plt.Normalize(vmin=water_vmin, vmax=water_vmax))
    water_sm._A = []

    carbon_cb = fig.colorbar(carbon_sm, ax=axes[:, 0], shrink=0.85, pad=0.01)
    carbon_cb.set_label("kg CO2eq / kWh")
    thermoelectric_water_cb = fig.colorbar(
        thermoelectric_water_sm, ax=axes[:, 1], shrink=0.85, pad=0.01
    )
    thermoelectric_water_cb.set_label("L / kWh")
    water_cb = fig.colorbar(water_sm, ax=axes[:, 2], shrink=0.85, pad=0.01)
    water_cb.set_label("log10(L / kWh)")
    water_cb.formatter = LOG10_TICK_FORMATTER
    water_cb.update_ticks()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_average_factor_map(
    factors: pd.DataFrame,
    geodf: gpd.GeoDataFrame,
    output_path: Path,
    id_col: str,
    level_label: str,
    start_year: int,
) -> None:
    carbon_vmax = float(factors["grid_carbon_factor_kgco2eq_per_kwh"].dropna().quantile(0.98))
    thermoelectric_water_vmax = float(
        factors["grid_thermoelectric_water_factor_l_per_kwh"].dropna().quantile(0.98)
    )
    total_water_log_values = transform_log10_positive(factors["grid_water_factor_l_per_kwh"])
    water_vmin = float(total_water_log_values.quantile(0.02))
    water_vmax = float(total_water_log_values.quantile(0.98))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    add_panel(
        ax=axes[0],
        geodf=geodf,
        values=factors[[id_col, "grid_carbon_factor_kgco2eq_per_kwh"]],
        id_col=id_col,
        year=start_year,
        value_column="grid_carbon_factor_kgco2eq_per_kwh",
        title=f"{level_label} avg grid carbon factor ({start_year}+)",
        cmap=CARBON_CMAP,
        vmin=0.0,
        vmax=carbon_vmax,
    )
    add_panel(
        ax=axes[1],
        geodf=geodf,
        values=factors[[id_col, "grid_thermoelectric_water_factor_l_per_kwh"]],
        id_col=id_col,
        year=start_year,
        value_column="grid_thermoelectric_water_factor_l_per_kwh",
        title=f"{level_label} avg grid thermoelectric water factor ({start_year}+)",
        cmap=WATER_CMAP,
        vmin=0.0,
        vmax=thermoelectric_water_vmax,
    )
    add_panel(
        ax=axes[2],
        geodf=geodf,
        values=factors[[id_col, "grid_water_factor_l_per_kwh"]].assign(
            grid_water_factor_log10_l_per_kwh=transform_log10_positive(
                factors["grid_water_factor_l_per_kwh"]
            )
        )[[id_col, "grid_water_factor_log10_l_per_kwh"]],
        id_col=id_col,
        year=start_year,
        value_column="grid_water_factor_log10_l_per_kwh",
        title=f"{level_label} avg grid water factor ({start_year}+)",
        cmap=WATER_CMAP,
        vmin=water_vmin,
        vmax=water_vmax,
    )

    carbon_sm = plt.cm.ScalarMappable(cmap=CARBON_CMAP, norm=plt.Normalize(vmin=0.0, vmax=carbon_vmax))
    carbon_sm._A = []
    thermoelectric_water_sm = plt.cm.ScalarMappable(
        cmap=WATER_CMAP, norm=plt.Normalize(vmin=0.0, vmax=thermoelectric_water_vmax)
    )
    thermoelectric_water_sm._A = []
    water_sm = plt.cm.ScalarMappable(cmap=WATER_CMAP, norm=plt.Normalize(vmin=water_vmin, vmax=water_vmax))
    water_sm._A = []

    carbon_cb = fig.colorbar(carbon_sm, ax=axes[0], shrink=0.85, pad=0.01)
    carbon_cb.set_label("kg CO2eq / kWh")
    thermoelectric_water_cb = fig.colorbar(
        thermoelectric_water_sm, ax=axes[1], shrink=0.85, pad=0.01
    )
    thermoelectric_water_cb.set_label("L / kWh")
    water_cb = fig.colorbar(water_sm, ax=axes[2], shrink=0.85, pad=0.01)
    water_cb.set_label("log10(L / kWh)")
    water_cb.formatter = LOG10_TICK_FORMATTER
    water_cb.update_ticks()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(case: str | Path) -> None:
    case_dir = Path(case)
    output_dir = case_dir / "outputs" / "carbon_water_factor"
    output_dir.mkdir(parents=True, exist_ok=True)

    ba_factors = load_ba_factor_data(case_dir)
    state_factors = aggregate_to_state(ba_factors)
    ba_avg_factors = average_factors_from_year(ba_factors, id_col="r", start_year=AVG_START_YEAR)
    state_avg_factors = average_factors_from_year(
        state_factors, id_col="st", start_year=AVG_START_YEAR
    )

    write_factor_outputs(ba_factors, output_dir, prefix="region")
    write_factor_outputs(state_factors, output_dir, prefix="state")
    write_average_factor_outputs(
        ba_avg_factors, output_dir, prefix="region", start_year=AVG_START_YEAR
    )
    write_average_factor_outputs(
        state_avg_factors, output_dir, prefix="state", start_year=AVG_START_YEAR
    )

    ba_map = get_map(case_dir, "r")
    state_map = get_map(case_dir, "st")

    plot_combined_factor_grid(
        factors=ba_factors,
        geodf=ba_map,
        output_path=output_dir / "region_grid_factor_maps_by_year.png",
        id_col="r",
        level_label="Region",
    )
    plot_combined_factor_grid(
        factors=state_factors,
        geodf=state_map,
        output_path=output_dir / "state_grid_factor_maps_by_year.png",
        id_col="st",
        level_label="State",
    )
    plot_average_factor_map(
        factors=ba_avg_factors,
        geodf=ba_map,
        output_path=output_dir / f"region_grid_factor_maps_avg_{AVG_START_YEAR}_onward.png",
        id_col="r",
        level_label="Region",
        start_year=AVG_START_YEAR,
    )
    plot_average_factor_map(
        factors=state_avg_factors,
        geodf=state_map,
        output_path=output_dir / f"state_grid_factor_maps_avg_{AVG_START_YEAR}_onward.png",
        id_col="st",
        level_label="State",
        start_year=AVG_START_YEAR,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate regional and state grid carbon and water factors and plot maps."
    )
    parser.add_argument("case", help="Path to ReEDS case directory")
    args = parser.parse_args()
    main(args.case)
