"""
Prepare data center load inputs for ReEDS.

The model optimizes IT load placement at the current ReEDS regional resolution
(BA or aggreg). PUE is applied after siting so that moving IT load changes the
total electric load.
"""

import argparse
import datetime
import logging
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import reeds


logger = logging.getLogger(__name__)


NUM_DATACENTER_SC_BINS = 60

# Once the 60-bin ascending-cost curve from ba_datacenter_supply_curve.csv is
# fully deployed, remaining endogenous/exogenous siting demand spills into this
# uncapped backstop bin. Its CAPEX is linearly extrapolated per region from the
# last two priced bins (dc59, dc60), continuing their cost step. GAMS excludes
# the bin from the per-bin capacity constraint (datacenter_sc_bin_uncapped), so
# the sentinel capacity below only supports ascending-cost bin-allocation
# bookkeeping (Python occupancy accounting and GAMS reporting) and is never
# itself a binding limit.
DATACENTER_SC_OVERFLOW_BIN = f"dc{NUM_DATACENTER_SC_BINS + 1}"
DATACENTER_SC_OVERFLOW_CAPACITY_MW = 1e7


def _read_modeled_years(inputs_case):
    years = pd.read_csv(os.path.join(inputs_case, "modeledyears.csv")).columns
    return sorted([int(y) for y in years])


def _read_state_yearly_metric(datacenter_dir, filename, value_name, states, years):
    """Read a wide state/province-by-year data-center metric file.

    Values before the first supplied year and after the last supplied year use
    the nearest supplied value. This keeps every modeled year defined when
    the source projection has a shorter horizon than the model.
    """
    path = os.path.join(datacenter_dir, filename)
    metric = pd.read_csv(path)
    if "st" not in metric.columns:
        raise ValueError(f"{filename} must contain st.")

    year_cols = [col for col in metric.columns if col != "st"]
    if not year_cols:
        raise ValueError(f"{filename} must contain at least one year column.")
    try:
        source_years = sorted(int(col) for col in year_cols)
    except ValueError as err:
        raise ValueError(f"{filename} year columns must be integer years.") from err
    if len(source_years) != len(year_cols):
        raise ValueError(f"{filename} contains duplicate year columns.")

    metric = metric.loc[metric["st"].isin(states), ["st"] + year_cols].copy()
    missing_states = sorted(set(states) - set(metric["st"]))
    if missing_states:
        raise ValueError(
            f"Missing {value_name} data for states/provinces: {missing_states}"
        )
    if metric["st"].duplicated().any():
        duplicates = sorted(metric.loc[metric["st"].duplicated(), "st"].unique())
        raise ValueError(f"Duplicate states/provinces in {filename}: {duplicates}")

    metric = metric.set_index("st")
    metric.columns = [int(col) for col in metric.columns]
    metric = metric.reindex(columns=source_years).apply(pd.to_numeric, errors="raise")
    target_years = sorted(set(source_years).union(years))
    metric = metric.reindex(columns=target_years).ffill(axis=1).bfill(axis=1)
    metric = metric.reindex(columns=years)
    if metric.isnull().any().any():
        raise ValueError(f"Missing {value_name} values in {filename}.")
    return (
        metric.rename_axis("*st")
        .rename_axis("t", axis="columns")
        .stack()
        .rename(value_name)
        .reset_index()
    )


def _get_region_column(hierarchy):
    for col in ["*r", "r", "ba"]:
        if col in hierarchy.columns:
            return col
    raise KeyError("hierarchy.csv must contain one of: *r, r, ba")


def _read_ba_supply_curve(reeds_path, datacenter_dir):
    """Read and validate the 60-step 2024-dollar CAPEX curve for every BA."""
    path = os.path.join(datacenter_dir, "ba_datacenter_supply_curve.csv")
    curve = pd.read_csv(path, dtype={"ba": str})
    required = {"ba", "bin", "capacity_mw", "capex_2024usd_per_mw"}
    missing = required - set(curve.columns)
    if missing:
        raise ValueError(
            "ba_datacenter_supply_curve.csv is missing columns: " f"{sorted(missing)}"
        )

    curve = curve.loc[:, ["ba", "bin", "capacity_mw", "capex_2024usd_per_mw"]].copy()
    curve["ba"] = curve["ba"].astype(str)
    for col in ["bin", "capacity_mw", "capex_2024usd_per_mw"]:
        curve[col] = pd.to_numeric(curve[col], errors="raise")
    if not np.all(np.isfinite(curve[["capacity_mw", "capex_2024usd_per_mw"]])):
        raise ValueError(
            "Data-center supply-curve capacities and CAPEX must be finite."
        )
    if not np.all(curve["bin"] == curve["bin"].astype(int)):
        raise ValueError(
            f"Data-center supply-curve bin values must be integers 1 through {NUM_DATACENTER_SC_BINS}."
        )
    curve["bin"] = curve["bin"].astype(int)
    if (curve["capacity_mw"] < 0).any():
        raise ValueError("Data-center supply-curve capacity_mw must be non-negative.")
    if (curve["capex_2024usd_per_mw"] < 0).any():
        raise ValueError("Data-center supply-curve CAPEX must be non-negative.")

    canonical_hierarchy = os.path.join(
        reeds_path, "inputs", "hierarchy_usacan_aggreg.csv"
    )
    hierarchy_path = (
        canonical_hierarchy
        if os.path.exists(canonical_hierarchy)
        else os.path.join(reeds_path, "inputs", "hierarchy.csv")
    )
    hierarchy = pd.read_csv(hierarchy_path, dtype={"ba": str})
    expected_bas = set(hierarchy["ba"].dropna().astype(str))
    supplied_bas = set(curve["ba"])
    if supplied_bas != expected_bas:
        missing_bas = sorted(expected_bas - supplied_bas)
        extra_bas = sorted(supplied_bas - expected_bas)
        raise ValueError(
            "ba_datacenter_supply_curve.csv must cover exactly the canonical BA set in "
            f"{os.path.basename(hierarchy_path)}; missing={missing_bas}, extra={extra_bas}"
        )

    expected_bins = set(range(1, NUM_DATACENTER_SC_BINS + 1))
    for ba, group in curve.groupby("ba", sort=False):
        if group["bin"].duplicated().any() or set(group["bin"]) != expected_bins:
            raise ValueError(
                f"BA {ba} must have each data-center supply-curve bin 1 through {NUM_DATACENTER_SC_BINS} exactly once."
            )
        ordered = group.sort_values("bin")
        if not np.all(np.diff(ordered["capex_2024usd_per_mw"].to_numpy()) > 0):
            raise ValueError(
                f"BA {ba} supply-curve CAPEX must strictly increase by bin."
            )
    return curve


def _read_2024_to_2004_deflator(inputs_case):
    """Read the 2024-to-2004 dollar-year deflator factor for a case."""
    path = os.path.join(inputs_case, "deflator.csv")
    deflator = pd.read_csv(path)
    deflator.columns = [str(col).lstrip("*") for col in deflator.columns]
    if not {"Dollar.Year", "Deflator"}.issubset(deflator.columns):
        raise ValueError("deflator.csv must contain *Dollar.Year and Deflator columns.")
    deflator["Dollar.Year"] = pd.to_numeric(deflator["Dollar.Year"], errors="raise")
    deflator["Deflator"] = pd.to_numeric(deflator["Deflator"], errors="raise")
    factors = deflator.set_index("Dollar.Year")["Deflator"]
    if 2024 not in factors.index:
        raise ValueError("deflator.csv must provide a 2024-to-2004 dollar deflator.")
    factor = factors.loc[2024]
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(
            "The 2024-to-2004 dollar deflator must be finite and positive."
        )
    return factor


def _convert_supply_curve_capex_to_2004(curve, inputs_case):
    """Add ReEDS' 2004-dollar CAPEX while preserving the source 2024-dollar cost."""
    factor = _read_2024_to_2004_deflator(inputs_case)
    converted = curve.copy()
    converted["capex_2004usd_per_mw"] = converted["capex_2024usd_per_mw"] * factor
    return converted


def _append_supply_curve_overflow_bin(supply_curve, inputs_case):
    """Add an uncapped, high-cost bin absorbing demand beyond the 60-bin curve.

    Once every region's ascending-cost bins (dc1-dc60) are exhausted, siting
    demand must still have somewhere to go rather than making the model
    infeasible. GAMS excludes this bin from the per-bin capacity constraint,
    so it behaves as an unlimited-capacity backstop. Its CAPEX is not a fixed
    input; it is linearly extrapolated per region from the last two priced
    bins (dc59, dc60), continuing their cost step.
    """
    factor = _read_2024_to_2004_deflator(inputs_case)
    regions = sorted(supply_curve["r"].unique())
    last_bin = f"dc{NUM_DATACENTER_SC_BINS}"
    second_last_bin = f"dc{NUM_DATACENTER_SC_BINS - 1}"
    last = (
        supply_curve.loc[supply_curve["dcbin"] == last_bin]
        .set_index("r")["capex_2024usd_per_mw"]
        .reindex(regions)
    )
    second_last = (
        supply_curve.loc[supply_curve["dcbin"] == second_last_bin]
        .set_index("r")["capex_2024usd_per_mw"]
        .reindex(regions)
    )
    if last.isnull().any() or second_last.isnull().any():
        raise ValueError(
            f"Every region must define {second_last_bin} and {last_bin} to "
            "extrapolate the uncapped overflow bin's CAPEX."
        )
    overflow_capex_2024 = 2.0 * last - second_last
    if (overflow_capex_2024 <= last).any():
        raise ValueError(
            "Extrapolated overflow-bin CAPEX must exceed the last priced "
            f"bin's ({last_bin}) CAPEX."
        )
    overflow = pd.DataFrame(
        {
            "r": regions,
            "dcbin": DATACENTER_SC_OVERFLOW_BIN,
            "capacity_mw": DATACENTER_SC_OVERFLOW_CAPACITY_MW,
            "capex_2024usd_per_mw": overflow_capex_2024.to_numpy(),
            "capex_2004usd_per_mw": overflow_capex_2024.to_numpy() * factor,
        }
    )
    return pd.concat([supply_curve, overflow], ignore_index=True)


def _aggregate_supply_curve_by_bin(segments):
    """Aggregate member BA capacities directly because every bin has one cost."""
    expected_bins = set(range(1, NUM_DATACENTER_SC_BINS + 1))
    grouped = segments.groupby("bin", sort=True)
    varying_cost = grouped["capex_2004usd_per_mw"].nunique()
    if (varying_cost > 1).any():
        invalid = varying_cost.loc[varying_cost > 1].index.tolist()
        raise ValueError(
            "All member BA supply curves must share one CAPEX value per bin; "
            f"inconsistent bins: {invalid}"
        )
    output = grouped.agg(
        capacity_mw=("capacity_mw", "sum"),
        capex_2024usd_per_mw=("capex_2024usd_per_mw", "first"),
        capex_2004usd_per_mw=("capex_2004usd_per_mw", "first"),
    ).reset_index()
    if set(output["bin"]) != expected_bins:
        raise ValueError(
            "Data-center supply-curve aggregation is missing one or more bins."
        )
    output.insert(0, "dcbin", output.pop("bin").map(lambda bin_id: f"dc{bin_id}"))
    return output


def _make_region_supply_curve(reeds_path, inputs_case, hierarchy, region_col):
    """Map BA curves to modeled regions and sum member capacities by bin."""
    curve = _read_ba_supply_curve(
        reeds_path, os.path.join(reeds_path, "inputs", "datacenter")
    )
    curve = _convert_supply_curve_capex_to_2004(curve, inputs_case)
    r_ba_path = os.path.join(inputs_case, "r_ba.csv")
    if os.path.exists(r_ba_path):
        r_ba = pd.read_csv(r_ba_path, dtype=str)
        if not {"r", "ba"}.issubset(r_ba.columns):
            raise ValueError("r_ba.csv must contain r and ba columns.")
    else:
        r_ba = pd.DataFrame({"r": curve["ba"].unique(), "ba": curve["ba"].unique()})
    valid_regions = set(hierarchy[region_col].astype(str))
    r_ba = r_ba.loc[r_ba["r"].isin(valid_regions), ["r", "ba"]].drop_duplicates()
    mapped = curve.merge(r_ba, on="ba", how="inner")
    if mapped.empty:
        raise ValueError(
            "No BA data-center supply-curve rows map to the current model regions."
        )

    output = []
    for region, group in mapped.groupby("r", sort=False):
        aggregated = _aggregate_supply_curve_by_bin(group)
        # This output is read by GAMS as a `table`, for which the first CSV
        # header is a dimension name, not a comment.  A leading `*` makes GAMS
        # skip the header and parse the first data row as column labels.
        aggregated.insert(0, "r", region)
        output.append(aggregated)
    return pd.concat(output, ignore_index=True).loc[
        :,
        [
            "r",
            "dcbin",
            "capacity_mw",
            "capex_2024usd_per_mw",
            "capex_2004usd_per_mw",
        ],
    ]


def _make_exogenous_supply_curve_requirement(
    cumulative,
    floor,
    ratios,
    supply_curve,
    layout_year,
    deployment_type,
    endogenous_share,
):
    """Region-level cumulative retained post-layout exogenous growth by year.

    This intentionally does *not* assign the requirement to specific
    supply-curve bins: GAMS fills bins for each region in ascending-cost order
    one model year at a time (see d1_datacenter_sc_fill.gms), using the bin
    capacity actually still available after that region's realized (solved)
    endogenous investment in prior years -- something this preprocessing step
    run before any solve has no way to know. Pre-assigning bins here (as an
    earlier version of this function did) can crowd out a bin's capacity with
    endogenous investment before a later year's exogenous requirement is
    known to need it, since the bin choice is fixed before solving starts.
    """
    columns = ["*r", "t", "MW"]
    if not 0.0 <= endogenous_share <= 1.0:
        raise ValueError("Sw_DatacenterEndogenousShare must be between 0 and 1.")
    if deployment_type == 0 or endogenous_share == 1.0:
        return pd.DataFrame(columns=columns)

    ratios = ratios.rename(columns={"*r": "r"}).copy()
    state_growth = cumulative.merge(
        floor, on=["st", "t"], how="left", suffixes=("_exog", "_floor")
    )
    if state_growth["MW_floor"].isnull().any():
        raise ValueError("Missing data-center layout floor for a modeled year.")
    state_growth["retained_exog_mw"] = (1.0 - endogenous_share) * (
        state_growth["MW_exog"] - state_growth["MW_floor"]
    )
    state_growth.loc[state_growth["t"] <= layout_year, "retained_exog_mw"] = 0.0
    if (state_growth["retained_exog_mw"] < -1e-6).any():
        raise ValueError(
            "Retained exogenous data-center growth cannot be negative after the "
            "layout cutoff year."
        )
    state_growth["retained_exog_mw"] = state_growth["retained_exog_mw"].clip(lower=0.0)

    regional = state_growth.merge(ratios, on="st", how="inner")
    regional["retained_exog_mw"] *= regional["ratio"]
    requirements = regional.groupby(["r", "t"], as_index=False)[
        "retained_exog_mw"
    ].sum()
    requirements = requirements.sort_values(["r", "t"])
    if requirements.groupby("r")["retained_exog_mw"].diff().lt(-1e-6).any():
        raise ValueError(
            "Cumulative retained exogenous data-center growth must not decrease."
        )
    required_regions = set(
        requirements.loc[requirements["retained_exog_mw"] > 1e-9, "r"]
    )
    curve_regions = set(supply_curve["r"])
    if required_regions - curve_regions:
        raise ValueError(
            "Missing data-center supply curves for regions with retained exogenous "
            f"growth: {sorted(required_regions - curve_regions)}"
        )
    total_capacity = supply_curve.groupby("r")["capacity_mw"].sum()
    over_capacity = requirements.merge(
        total_capacity.rename("total_capacity_mw"), on="r", how="left"
    )
    if (
        over_capacity["retained_exog_mw"] > over_capacity["total_capacity_mw"] + 1e-6
    ).any():
        raise ValueError(
            "Retained exogenous data-center growth exceeds total supply-curve "
            "capacity (including the uncapped overflow bin) for at least one region."
        )

    return requirements.rename(columns={"r": "*r", "retained_exog_mw": "MW"})[columns]


def _normalize_scenario(raw_scenario):
    scenario = str(raw_scenario).strip().lower()
    if scenario not in {"low", "mid", "high"}:
        raise ValueError(
            f"Unsupported Sw_DatacenterScenario={raw_scenario}. "
            "Expected one of: high, mid, low."
        )
    return scenario


def _make_cumulative_it(incremental, states, years):
    return _make_cumulative_metric(incremental, states, years, "it_load_gw")


def _make_cumulative_metric(incremental, states, years, value_col):
    year_min = min(min(years), int(incremental["year"].min()))
    year_max = max(max(years), int(incremental["year"].max()))
    full_index = pd.MultiIndex.from_product(
        [states, range(year_min, year_max + 1)], names=["st", "year"]
    )
    inc = (
        incremental.rename(columns={"region": "st"})
        .groupby(["st", "year"], as_index=True)[value_col]
        .sum()
        .mul(1000.0)
        .reindex(full_index, fill_value=0.0)
        .groupby(level="st")
        .cumsum()
        .rename("MW")
        .reset_index()
    )
    inc = inc[inc["year"].isin(years)].rename(columns={"year": "t"})
    return inc


def _make_incremental_floor(incremental, states, years, layout_year):
    cutoff_year = layout_year
    prelayout = incremental.loc[incremental["year"] <= cutoff_year].copy()
    if prelayout.empty:
        floor = pd.Series(0.0, index=pd.Index(states, name="st"), name="MW")
    else:
        floor = (
            prelayout.rename(columns={"region": "st"})
            .groupby("st")["it_load_gw"]
            .sum()
            .mul(1000.0)
            .reindex(states, fill_value=0.0)
            .rename("MW")
        )
    return (
        floor.reset_index()
        .merge(pd.DataFrame({"t": years}), how="cross")
        .loc[:, ["st", "t", "MW"]]
    )


def _read_allh(inputs_case):
    return (
        pd.read_csv(
            os.path.join(inputs_case, "set_allh.csv"), header=None, names=["h"]
        )["h"]
        .astype(str)
        .tolist()
    )


def _select_temporal_directories(inputs_case):
    """Return rep plus the latest stress iteration available for each solve year.

    Stress-period directories are cumulative within a solve year.  Reading both
    ``stress{year}i0`` and ``stress{year}i1`` would therefore map the same model
    timeslices into multiple data-center days.  Keep only the newest iteration
    for each year, while retaining legacy stress directory names such as
    ``stress0`` used by PCM workflows.
    """
    directories = [
        dirname
        for dirname in os.listdir(inputs_case)
        if os.path.isdir(os.path.join(inputs_case, dirname))
    ]
    selected = ["rep"] if "rep" in directories else []
    latest = {}
    legacy = []
    for dirname in directories:
        if not dirname.startswith("stress"):
            continue
        match = re.fullmatch(r"stress(?P<year>\d+)i(?P<iteration>\d+)", dirname)
        if match is None:
            legacy.append(dirname)
            continue
        year = int(match.group("year"))
        iteration = int(match.group("iteration"))
        if year not in latest or iteration > latest[year][0]:
            latest[year] = (iteration, dirname)
    selected.extend(sorted(legacy))
    selected.extend(latest[year][1] for year in sorted(latest))
    return selected


def _read_timeslice_hours(inputs_case, sw, hours):
    rep_duration = float(sw.get("GSw_HourlyChunkLengthRep", 1))
    stress_duration = float(sw.get("GSw_HourlyChunkLengthStress", rep_duration))
    timeslice_hours = {h: rep_duration for h in hours}

    rep_set = os.path.join(inputs_case, "rep", "set_h.csv")
    if os.path.exists(rep_set):
        rep_hours = pd.read_csv(rep_set).iloc[:, 0].astype(str).tolist()
        for h in rep_hours:
            timeslice_hours[h] = rep_duration

    for dirname in _select_temporal_directories(inputs_case):
        stress_set = os.path.join(inputs_case, dirname, "set_h.csv")
        if dirname.startswith("stress") and os.path.exists(stress_set):
            stress_hours = pd.read_csv(stress_set).iloc[:, 0].astype(str).tolist()
            for h in stress_hours:
                timeslice_hours[h] = stress_duration

    return pd.Series(timeslice_hours, name="duration_hours")


def _validate_flex_switches(
    train_temporal_share,
    train_max_delay_hours,
    spatial_scope,
    spatial_share,
    move_penalty,
):
    """Validate the task-specific data-center flexibility switches."""
    for name, value in {
        "Sw_DatacenterTrainTemporalFlexShare": train_temporal_share,
        "Sw_DatacenterSpatialFlexShare": spatial_share,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    if (
        train_max_delay_hours < 0.0
        or train_max_delay_hours >= 24.0
        or not float(train_max_delay_hours).is_integer()
    ):
        raise ValueError(
            "Sw_DatacenterTrainMaxDelayHours must be an integer between 0 and 23."
        )
    if spatial_scope not in {0, 1, 2}:
        raise ValueError("Sw_DatacenterSpatialFlexScope must be one of 0, 1, or 2.")
    if move_penalty < 0.0:
        raise ValueError("Sw_DatacenterMovePenalty must be non-negative.")


def _validate_deployment_type(raw_deployment_type):
    """Return a validated integer data-center deployment mode."""
    try:
        deployment_type = float(raw_deployment_type)
    except (TypeError, ValueError) as err:
        raise ValueError(
            "Sw_DatacenterDeploymentType must be one of 0, 1, or 2."
        ) from err
    if not np.isfinite(deployment_type) or not deployment_type.is_integer():
        raise ValueError("Sw_DatacenterDeploymentType must be one of 0, 1, or 2.")
    deployment_type = int(deployment_type)
    if deployment_type not in {0, 1, 2}:
        raise ValueError("Sw_DatacenterDeploymentType must be one of 0, 1, or 2.")
    return deployment_type


def _make_train_daily_inputs(inputs_case, sw, regions):
    """Create training days and sparse, direct temporal-routing inputs.

    A model timeslice is an average-MW block.  Its actual duration comes from
    the same representative/stress chunk switches used by GAMS.  Spatial
    allocation is per representative-day occurrence, while day_weight restores
    annual MWh and movement cost weighting. Representative days are independent
    24-hour cycles; each stress period is one cycle across all of its days.
    """
    all_hours = _read_allh(inputs_case)
    duration = _read_timeslice_hours(inputs_case, sw, all_hours)
    max_delay = float(sw.get("Sw_DatacenterTrainMaxDelayHours", 3))
    if not all(np.isclose(24.0 / value, round(24.0 / value)) for value in duration):
        raise ValueError("Data-center timeslice durations must divide 24 hours.")

    day_rows, day_hour_rows, weight_rows = [], [], []
    route_source_rows, route_destination_rows = [], []
    route_rep_wrap_rows, route_stress_cross_day_rows = [], []
    route_stress_wrap_rows, route_delay_rows = [], []
    route_index = 0
    day_by_hours = {}

    def add_routes(cycle_hours, cycle_kind, hour_to_day):
        """Add direct source-to-destination routes for one cyclic time domain."""
        nonlocal route_index
        cycle_durations = duration.reindex(cycle_hours)
        cycle_ends = cycle_durations.cumsum().to_numpy()
        cycle_length = float(cycle_ends[-1])
        for source_index, source_hour in enumerate(cycle_hours):
            source_end = cycle_ends[source_index]
            for destination_index, destination_hour in enumerate(cycle_hours):
                destination_end = cycle_ends[destination_index]
                if destination_index < source_index:
                    destination_end += cycle_length
                delay_hours = destination_end - source_end
                if delay_hours < -1e-9 or delay_hours > max_delay + 1e-9:
                    continue
                route_index += 1
                route = f"dctr{route_index:07d}"
                route_source_rows.append({"route": route, "h": source_hour})
                route_destination_rows.append({"route": route, "h": destination_hour})
                route_delay_rows.append(
                    {"route": route, "delay_hours": max(0.0, delay_hours)}
                )
                wraps_cycle = destination_index < source_index
                if cycle_kind == "rep" and wraps_cycle:
                    route_rep_wrap_rows.append({"route": route})
                if cycle_kind == "stress":
                    if hour_to_day[source_hour] != hour_to_day[destination_hour]:
                        route_stress_cross_day_rows.append({"route": route})
                    if wraps_cycle:
                        route_stress_wrap_rows.append({"route": route})

    for dirname in _select_temporal_directories(inputs_case):
        directory = os.path.join(inputs_case, dirname)
        required = {
            "h_szn.csv": os.path.join(directory, "h_szn.csv"),
            "h_szn_start.csv": os.path.join(directory, "h_szn_start.csv"),
            "h_szn_end.csv": os.path.join(directory, "h_szn_end.csv"),
            "nexth.csv": os.path.join(directory, "nexth.csv"),
            "numhours.csv": os.path.join(directory, "numhours.csv"),
        }
        present = [os.path.exists(path) for path in required.values()]
        if not any(present):
            continue
        if not all(present):
            missing = [
                name for name, path in required.items() if not os.path.exists(path)
            ]
            raise FileNotFoundError(
                f"{dirname} is missing daily-training inputs: {missing}"
            )

        h_szn = pd.read_csv(required["h_szn.csv"], dtype=str).rename(
            columns={"*h": "h", "season": "period"}
        )
        starts = pd.read_csv(required["h_szn_start.csv"], dtype=str).rename(
            columns={"*season": "period"}
        )
        ends = pd.read_csv(required["h_szn_end.csv"], dtype=str).rename(
            columns={"*season": "period"}
        )
        nexth = pd.read_csv(required["nexth.csv"], dtype=str).rename(
            columns={"*h": "h", "h": "hh"}
        )
        numhours = pd.read_csv(required["numhours.csv"]).rename(columns={"*h": "h"})
        if (
            not {"h", "period"}.issubset(h_szn)
            or not {"period", "h"}.issubset(starts)
            or not {"period", "h"}.issubset(ends)
            or not {"h", "hh"}.issubset(nexth)
            or not {"h", "numhours"}.issubset(numhours)
        ):
            raise ValueError(f"{dirname} daily-training inputs have invalid columns.")
        h_szn = h_szn.loc[:, ["h", "period"]].drop_duplicates()
        starts = starts.loc[:, ["period", "h"]].drop_duplicates()
        ends = ends.loc[:, ["period", "h"]].drop_duplicates()
        if (
            h_szn["h"].duplicated().any()
            or starts["period"].duplicated().any()
            or ends["period"].duplicated().any()
        ):
            raise ValueError(f"{dirname} has ambiguous daily-training period mappings.")
        if set(starts["period"]) != set(h_szn["period"]) or set(ends["period"]) != set(
            h_szn["period"]
        ):
            raise ValueError(
                f"{dirname} must define one start and end for every period."
            )
        next_map = dict(nexth.loc[:, ["h", "hh"]].itertuples(index=False, name=None))
        annual_hours = pd.to_numeric(
            numhours.set_index("h")["numhours"], errors="raise"
        )

        for period, start in starts.itertuples(index=False, name=None):
            period_hours = set(h_szn.loc[h_szn["period"] == period, "h"])
            end = ends.loc[ends["period"] == period, "h"].iloc[0]
            sequence, current = [], start
            while True:
                if current not in period_hours or current in sequence:
                    raise ValueError(
                        f"{dirname}/{period} does not form a non-wrapping hour sequence."
                    )
                sequence.append(current)
                if current == end:
                    break
                if current not in next_map:
                    raise ValueError(
                        f"{dirname}/{period} has no successor before its end timeslice."
                    )
                current = next_map[current]
            if set(sequence) != period_hours:
                raise ValueError(
                    f"{dirname}/{period} sequence does not cover all of its timeslices."
                )
            durations = duration.reindex(sequence)
            if durations.isnull().any() or not np.isclose(durations.sum() % 24.0, 0.0):
                raise ValueError(
                    f"{dirname}/{period} must contain an integer number of 24-hour days."
                )

            elapsed, day_index, day_hours = 0.0, 1, []
            period_days = []
            for hour in sequence:
                block = float(durations.loc[hour])
                if elapsed + block > 24.0 + 1e-9:
                    raise ValueError(
                        f"{dirname}/{period} timeslices cannot be split across model days."
                    )
                day_hours.append(hour)
                elapsed += block
                if np.isclose(elapsed, 24.0):
                    safe_period = re.sub(r"[^A-Za-z0-9_]", "_", str(period))
                    day = f"dc_{dirname}_{safe_period}_{day_index:03d}"
                    occurrences = annual_hours.reindex(day_hours) / durations.reindex(
                        day_hours
                    )
                    if occurrences.isnull().any() or not np.allclose(
                        occurrences, occurrences.iloc[0]
                    ):
                        raise ValueError(
                            f"{dirname}/{period} has inconsistent annual weights within a model day."
                        )
                    weight = float(occurrences.iloc[0])
                    day_key = tuple(day_hours)
                    if day_key in day_by_hours:
                        existing_day, existing_weight = day_by_hours[day_key]
                        if not np.isclose(weight, existing_weight):
                            logger.info(
                                "Skipping duplicate data-center training day %s/%s "
                                "(weight=%s); retaining %s (weight=%s).",
                                dirname,
                                period,
                                weight,
                                existing_day,
                                existing_weight,
                            )
                        elapsed, day_hours = 0.0, []
                        day_index += 1
                        continue
                    day_rows.append({"day": day})
                    weight_rows.append({"day": day, "weight": weight})
                    day_hour_rows.extend({"day": day, "h": h} for h in day_hours)
                    day_by_hours[day_key] = (day, weight)
                    period_days.append((day, list(day_hours)))
                    elapsed, day_hours = 0.0, []
                    day_index += 1
            if day_hours:
                raise ValueError(
                    f"{dirname}/{period} ends with an incomplete model day."
                )
            if dirname == "rep":
                for day, cycle_hours in period_days:
                    add_routes(cycle_hours, "rep", {hour: day for hour in cycle_hours})
            elif period_days:
                cycle_hours = [hour for _, hours in period_days for hour in hours]
                hour_to_day = {
                    hour: day for day, hours in period_days for hour in hours
                }
                add_routes(cycle_hours, "stress", hour_to_day)

    if not day_rows:
        raise FileNotFoundError(
            "No representative or stress model days found for data-center training."
        )
    day_hours = pd.DataFrame(day_hour_rows)
    if day_hours["h"].duplicated().any():
        duplicates = sorted(day_hours.loc[day_hours["h"].duplicated(), "h"].unique())
        raise ValueError(
            "Data-center timeslices map to multiple model days: " f"{duplicates}"
        )
    shares = day_hours.merge(
        duration.rename("duration_hours"), left_on="h", right_index=True, how="left"
    )
    shares["share"] = shares["duration_hours"] / shares.groupby("day")[
        "duration_hours"
    ].transform("sum")
    regions = sorted(set(map(str, regions)))
    intraday = pd.concat(
        [
            shares.assign(**{"*r": region})[["*r", "day", "h", "share"]]
            for region in regions
        ],
        ignore_index=True,
    )
    return {
        "days": pd.DataFrame(day_rows).drop_duplicates(),
        "day_hours": day_hours,
        "weights": pd.DataFrame(weight_rows),
        "shares": intraday,
        "routes": pd.DataFrame(route_delay_rows, columns=["route"]),
        "route_source": pd.DataFrame(route_source_rows, columns=["route", "h"]),
        "route_destination": pd.DataFrame(
            route_destination_rows, columns=["route", "h"]
        ),
        "route_rep_wrap": pd.DataFrame(route_rep_wrap_rows, columns=["route"]),
        "route_stress_cross_day": pd.DataFrame(
            route_stress_cross_day_rows, columns=["route"]
        ),
        "route_stress_wrap": pd.DataFrame(route_stress_wrap_rows, columns=["route"]),
        "route_delay": pd.DataFrame(route_delay_rows, columns=["route", "delay_hours"]),
    }


def _read_ba_distance(reeds_path):
    """Read the BA shortest-path matrix used by transmission preprocessing."""
    path = os.path.join(
        reeds_path, "inputs", "transmission", "transmission_distance_ba.h5"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing data-center distance proxy: {path}")
    with h5py.File(path, "r") as h5_file:
        required = {"index_0", "columns", "data"}
        missing = required - set(h5_file)
        if missing:
            raise ValueError(f"{path} is missing matrix datasets: {sorted(missing)}")
        decode = lambda item: item.decode() if isinstance(item, bytes) else str(item)
        distance = pd.DataFrame(
            h5_file["data"][:],
            index=[decode(item) for item in h5_file["index_0"][:]],
            columns=[decode(item) for item in h5_file["columns"][:]],
        )
    return distance.apply(pd.to_numeric, errors="raise")


def _make_spatial_flex_inputs(reeds_path, inputs_case, hierarchy, region_col, scope):
    """Return candidate routes and BA-anchor shortest-path distances in km.

    GAMS filters these hierarchy-wide candidates to regions in datacenter_r after
    the siting switches and data-center input availability have been evaluated.
    """
    regions = (
        hierarchy.loc[:, [region_col, "country"]]
        .drop_duplicates()
        .rename(columns={region_col: "r"})
    )
    regions["r"] = regions["r"].astype(str)
    if regions["r"].duplicated().any() or regions["country"].isnull().any():
        raise ValueError(
            "Each current data-center region must have exactly one country."
        )
    distance_ba = _read_ba_distance(reeds_path)
    anchors = pd.Series(index=regions["r"], dtype=object)
    anchor_path = os.path.join(inputs_case, "aggreg2anchorreg.csv")
    if os.path.exists(anchor_path):
        anchor_df = pd.read_csv(anchor_path, dtype=str)
        if not {"aggreg", "rb"}.issubset(anchor_df):
            raise ValueError("aggreg2anchorreg.csv must contain aggreg and rb columns.")
        if anchor_df["aggreg"].duplicated().any():
            raise ValueError(
                "aggreg2anchorreg.csv must have one anchor BA per aggregate region."
            )
        anchors.update(anchor_df.set_index("aggreg")["rb"])
    direct = anchors.isnull() & anchors.index.isin(distance_ba.index)
    anchors.loc[direct] = anchors.index[direct]
    missing = sorted(anchors[anchors.isnull()].index.tolist())
    if missing:
        raise ValueError(
            "Data-center regions cannot be mapped to a unique BA anchor: " f"{missing}"
        )
    unknown = sorted(set(anchors) - set(distance_ba.index))
    if unknown:
        raise ValueError(
            f"Data-center BA anchors are absent from transmission distances: {unknown}"
        )

    lookup_country = regions.set_index("r")["country"]
    route_rows = []
    distance_rows = []
    for origin in regions["r"]:
        for destination in regions["r"]:
            allowed = (
                scope == 2
                or (
                    scope == 1 and lookup_country[origin] == lookup_country[destination]
                )
                or (scope == 0 and origin == destination)
            )
            if not allowed:
                continue
            if origin == destination:
                miles = 0.0
            else:
                try:
                    miles = float(
                        distance_ba.loc[anchors[origin], anchors[destination]]
                    )
                except KeyError as err:
                    raise ValueError(
                        f"Missing transmission-distance route for {origin}->{destination} "
                        f"({anchors[origin]}->{anchors[destination]})."
                    ) from err
                if not np.isfinite(miles) or miles < 0.0:
                    raise ValueError(
                        f"Invalid transmission distance for {origin}->{destination}."
                    )
            route_rows.append({"r": origin, "rr": destination})
            distance_rows.append(
                {"*r": origin, "rr": destination, "km": miles * 1.609344}
            )
    return pd.DataFrame(route_rows), pd.DataFrame(distance_rows)


def _read_train_infer_utilization(datacenter_dir, years):
    """Read annual workload parameters, expressed as fractions rather than percent."""
    filename = "yearly_train_infer_util.csv"
    path = os.path.join(datacenter_dir, filename)
    required = {"year", "train_share", "infer_share", "train_util", "infer_util"}
    utilization = pd.read_csv(path)
    missing = required - set(utilization.columns)
    if missing:
        raise ValueError(f"{filename} is missing required columns: {sorted(missing)}")
    if utilization["year"].duplicated().any():
        duplicates = sorted(
            utilization.loc[utilization["year"].duplicated(), "year"].unique()
        )
        raise ValueError(f"{filename} contains duplicate years: {duplicates}")

    utilization = utilization.loc[
        :, ["year", "train_share", "infer_share", "train_util", "infer_util"]
    ].copy()
    utilization["year"] = pd.to_numeric(utilization["year"], errors="raise").astype(int)
    parameter_cols = ["train_share", "infer_share", "train_util", "infer_util"]
    utilization[parameter_cols] = utilization[parameter_cols].apply(
        pd.to_numeric, errors="raise"
    )
    if (
        ((utilization[parameter_cols] < 0.0) | (utilization[parameter_cols] > 1.0))
        .any()
        .any()
    ):
        raise ValueError(
            f"{filename} workload shares and utilisations must be fractions in [0, 1]."
        )
    if not np.allclose(utilization["train_share"] + utilization["infer_share"], 1.0):
        raise ValueError(
            f"{filename} train_share and infer_share must sum to 1 for every year."
        )

    utilization = utilization.set_index("year").sort_index()
    target_years = sorted(set(utilization.index).union(years))
    utilization = utilization.reindex(target_years).ffill().bfill().reindex(years)
    if utilization.isnull().any().any():
        raise ValueError(
            f"{filename} does not define every modeled year after endpoint filling."
        )
    return (
        utilization.rename_axis("*t")
        .reset_index()
        .melt(id_vars="*t", var_name="workload_param", value_name="value")
    )


def _read_inference_trace(datacenter_dir):
    filename = "hourly_inference_trace.csv"
    trace = pd.read_csv(os.path.join(datacenter_dir, filename))
    required = {"type", "h", "multiplier"}
    missing = required - set(trace.columns)
    if missing:
        raise ValueError(f"{filename} is missing required columns: {sorted(missing)}")
    trace = trace.loc[:, ["type", "h", "multiplier"]].copy()
    trace["type"] = trace["type"].astype(str).str.strip().str.lower()
    trace["h"] = pd.to_numeric(trace["h"], errors="raise").astype(int)
    trace["multiplier"] = pd.to_numeric(trace["multiplier"], errors="raise")
    expected = pd.MultiIndex.from_product(
        [["wd", "we"], range(24)], names=["type", "h"]
    )
    if trace.duplicated(["type", "h"]).any() or set(trace["type"]) != {"wd", "we"}:
        raise ValueError(
            f"{filename} must contain one wd and one we row for each hour 0--23."
        )
    if not trace.set_index(["type", "h"]).index.equals(expected):
        trace = trace.set_index(["type", "h"]).reindex(expected)
        if trace["multiplier"].isnull().any():
            raise ValueError(
                f"{filename} must contain one wd and one we row for each hour 0--23."
            )
        trace = trace.reset_index()
    if (trace["multiplier"] < 0.0).any():
        raise ValueError(f"{filename} multipliers must be non-negative.")
    weekly_total = (
        5.0 * trace.loc[trace["type"] == "wd", "multiplier"].sum()
        + 2.0 * trace.loc[trace["type"] == "we", "multiplier"].sum()
    )
    if not np.isclose(weekly_total, 24.0 * 7.0, atol=1e-7):
        raise ValueError(
            f"{filename} must satisfy 5*sum(wd) + 2*sum(we) = 168 "
            f"so its representative-week hourly mean is 1; found {weekly_total}."
        )
    return trace


def _make_region_timezones(datacenter_dir, inputs_case, hierarchy, region_col, states):
    """Assign one fixed UTC offset to each current ReEDS region from BA inputs."""
    timezone = pd.read_csv(os.path.join(datacenter_dir, "ba_timezone.csv"))
    if not {"ba", "timezone"}.issubset(timezone.columns):
        raise ValueError("ba_timezone.csv must contain ba and timezone columns.")
    timezone = timezone.loc[:, ["ba", "timezone"]].copy()
    timezone["ba"] = timezone["ba"].astype(str)
    timezone["timezone"] = pd.to_numeric(timezone["timezone"], errors="raise")
    if timezone["ba"].duplicated().any():
        raise ValueError("ba_timezone.csv must contain one row per BA.")
    if (
        (timezone["timezone"] % 1 != 0)
        | (timezone["timezone"] < -12)
        | (timezone["timezone"] > 14)
    ).any():
        raise ValueError(
            "ba_timezone.csv timezone must be an integer UTC offset between -12 and 14."
        )

    ba_weights = pd.read_csv(os.path.join(datacenter_dir, "exog_ba_st_dc_ratio.csv"))
    ba_weights = (
        ba_weights.loc[ba_weights["st"].isin(states)]
        .groupby("ba", as_index=False)["ratio"]
        .sum()
    )
    r_ba_path = os.path.join(inputs_case, "r_ba.csv")
    if os.path.exists(r_ba_path):
        r_ba = pd.read_csv(r_ba_path, dtype=str)
        if not {"r", "ba"}.issubset(r_ba.columns):
            raise ValueError("r_ba.csv must contain r and ba columns.")
    else:
        r_ba = pd.DataFrame({"r": timezone["ba"], "ba": timezone["ba"]})
    valid_regions = pd.DataFrame({"r": hierarchy[region_col].astype(str).unique()})
    mapped = (
        valid_regions.merge(r_ba.loc[:, ["r", "ba"]], on="r", how="left")
        .merge(timezone, on="ba", how="left")
        .merge(ba_weights, on="ba", how="left")
    )
    if mapped[["ba", "timezone"]].isnull().any().any():
        missing = (
            mapped.loc[mapped["ba"].isnull() | mapped["timezone"].isnull(), "r"]
            .unique()
            .tolist()
        )
        raise ValueError(f"Missing BA timezone mapping for regions: {sorted(missing)}")
    mapped["ratio"] = mapped["ratio"].fillna(0.0)
    selected = []
    for r, group in mapped.groupby("r", sort=False):
        if len(group) == 1:
            selected.append({"*r": r, "timezone": int(group["timezone"].iloc[0])})
            continue
        max_weight = group["ratio"].max()
        if max_weight <= 0.0:
            if group["timezone"].nunique() != 1:
                raise ValueError(
                    f"No data-center BA allocation weight is available to select a timezone for {r}, "
                    "and its BAs span multiple timezones."
                )
            selected.append({"*r": r, "timezone": int(group["timezone"].iloc[0])})
            continue
        dominant = group.loc[np.isclose(group["ratio"], max_weight)]
        if dominant["timezone"].nunique() != 1:
            raise ValueError(
                f"Data-center BA allocation weights tie across timezones for {r}."
            )
        selected.append({"*r": r, "timezone": int(dominant["timezone"].iloc[0])})
    return pd.DataFrame(selected)


def _read_case_hour_maps(inputs_case):
    frames = []
    set_hours = []
    for dirname in _select_temporal_directories(inputs_case):
        path = os.path.join(inputs_case, dirname, "hmap_myr.csv")
        if os.path.exists(path):
            hmap = pd.read_csv(path)
            timestamp_col = (
                "*timestamp" if "*timestamp" in hmap.columns else "timestamp"
            )
            if timestamp_col not in hmap.columns or "h" not in hmap.columns:
                raise ValueError(f"{path} must contain timestamp and h columns.")
            if not hmap.empty:
                hmap = hmap.loc[:, [timestamp_col, "h"]].rename(
                    columns={timestamp_col: "timestamp"}
                )
                hmap["h"] = hmap["h"].astype(str)
                timestamps = pd.to_datetime(hmap["timestamp"], errors="raise")
                if timestamps.dt.tz is None:
                    timestamps = timestamps.dt.tz_localize("Etc/GMT+6")
                hmap["timestamp"] = timestamps.dt.tz_convert("UTC")
                frames.append(hmap)
        set_h_path = os.path.join(inputs_case, dirname, "set_h.csv")
        if os.path.exists(set_h_path):
            set_hours.extend(pd.read_csv(set_h_path).iloc[:, 0].astype(str).tolist())
    if not frames:
        mapped = pd.DataFrame(columns=["timestamp", "h"])
    else:
        mapped = pd.concat(frames, ignore_index=True)
    missing_hours = sorted(set(set_hours) - set(mapped["h"]))
    if missing_hours:
        try:
            fallback = pd.DataFrame(
                {
                    "h": missing_hours,
                    "timestamp": [
                        reeds.timeseries.h2timestamp(hour).tz_convert("UTC")
                        for hour in missing_hours
                    ],
                }
            )
        except (TypeError, ValueError) as err:
            raise ValueError(
                "Could not recover timestamps for timeslices missing from hmap_myr.csv."
            ) from err
        mapped = pd.concat([mapped, fallback], ignore_index=True)
    if mapped.empty:
        raise FileNotFoundError(
            "No representative or stress timeslices found for data-center inference trace."
        )
    return mapped


def _make_inference_multiplier(inputs_case, region_timezones, trace):
    """Map the weekday inference trace to every modeled timeslice.

    All representative and stress timeslices deliberately use the weekday curve
    as a conservative demand stress test; the local timezone still determines
    the applicable hour of that curve.
    """
    hmap = _read_case_hour_maps(inputs_case)
    trace_lookup = trace.set_index(["type", "h"])["multiplier"]
    output = []
    for region, timezone in region_timezones[["*r", "timezone"]].itertuples(
        index=False, name=None
    ):
        local = hmap["timestamp"] + pd.to_timedelta(timezone, unit="h")
        values = pd.Series(
            [trace_lookup[("wd", hour)] for hour in local.dt.hour],
            index=hmap.index,
        )
        output.append(
            pd.DataFrame({"*r": region, "h": hmap["h"], "multiplier": values})
            .groupby(["*r", "h"], as_index=False)["multiplier"]
            .mean()
        )
    return pd.concat(output, ignore_index=True)


def _make_region_state_ratios(
    datacenter_dir, inputs_case, hierarchy, region_col, states
):
    ratios = pd.read_csv(os.path.join(datacenter_dir, "exog_ba_st_dc_ratio.csv"))
    ratios["ba"] = ratios["ba"].astype(str)

    r_ba_path = os.path.join(inputs_case, "r_ba.csv")
    if os.path.exists(r_ba_path):
        r_ba = pd.read_csv(r_ba_path)
        r_ba["ba"] = r_ba["ba"].astype(str)
        r_ba["r"] = r_ba["r"].astype(str)
        ratios = ratios.merge(r_ba, on="ba", how="inner")
    else:
        ratios["r"] = ratios["ba"]

    valid_regions = set(hierarchy[region_col].astype(str))
    ratios = (
        ratios.loc[
            ratios["r"].isin(valid_regions) & ratios["st"].isin(states),
            ["r", "st", "ratio"],
        ]
        .groupby(["r", "st"], as_index=False)["ratio"]
        .sum()
        .rename(columns={"r": "*r"})
    )
    _validate_ratio_sums(
        ratios.rename(columns={"*r": "r"}),
        ["st"],
        ["ratio"],
        "exog_ba_st_dc_ratio.csv",
    )
    return ratios


def _validate_ratio_sums(df, group_cols, value_cols, label, tol=1e-5):
    sums = df.groupby(group_cols, as_index=False)[value_cols].sum()
    for col in value_cols:
        bad = sums.loc[~np.isclose(sums[col], 1.0, atol=tol), group_cols + [col]]
        if not bad.empty:
            raise ValueError(
                f"{label} must sum to 1 within tolerance for {group_cols}. "
                f"Invalid rows for {col}: {bad.to_dict('records')}"
            )


def _read_country_increments(datacenter_dir, scenario):
    path = os.path.join(datacenter_dir, "country_annual_it_cap_increment.csv")
    increments = pd.read_csv(path)
    required_cols = {"scenario", "country"}
    if not required_cols.issubset(increments.columns):
        raise ValueError(
            "country_annual_it_cap_increment.csv must contain scenario and country."
        )

    increments["scenario"] = increments["scenario"].map(_normalize_scenario)
    supported_countries = {"USA", "CAN"}
    found = set(increments["country"])
    if found != supported_countries:
        raise ValueError(
            "country_annual_it_cap_increment.csv must contain exactly USA and CAN. "
            f"Found: {sorted(found)}"
        )

    supported_scenarios = {"low", "mid", "high"}
    found_scenarios = set(increments["scenario"])
    if found_scenarios != supported_scenarios:
        raise ValueError(
            "country_annual_it_cap_increment.csv must contain exactly low, mid, high. "
            f"Found: {sorted(found_scenarios)}"
        )

    increments = increments.loc[increments["scenario"] == scenario].copy()
    year_cols = sorted(
        [col for col in increments.columns if col not in ["scenario", "country"]],
        key=int,
    )
    increments[year_cols] = increments[year_cols].astype(float)
    return increments, year_cols


def _read_exogenous_allocation(datacenter_dir, states, max_required_year):
    path = os.path.join(
        datacenter_dir, "exog_st_country_annual_increment_allocation_ratio.csv"
    )
    exog = pd.read_csv(path)
    required_cols = {"st", "country"}
    if not required_cols.issubset(exog.columns):
        raise ValueError(
            "exog_st_country_annual_increment_allocation_ratio.csv must contain st and country."
        )

    year_cols = sorted(
        [col for col in exog.columns if col not in ["st", "country"]], key=int
    )
    exog[year_cols] = exog[year_cols].astype(float)
    exog = exog.loc[exog["st"].isin(states)].copy()
    missing_states = sorted(set(states) - set(exog["st"]))
    if missing_states:
        raise ValueError(
            "Missing exogenous allocation data for states/provinces: "
            f"{missing_states}"
        )

    needed_years = [
        str(y) for y in range(min(map(int, year_cols)), int(max_required_year) + 1)
    ]
    for year in needed_years:
        if year not in exog.columns:
            exog[year] = np.nan
    ordered_years = sorted(
        [col for col in exog.columns if col not in ["st", "country"]], key=int
    )
    exog = exog.set_index(["st", "country"])[ordered_years].T.ffill().T.reset_index()
    if exog[needed_years].isnull().any().any():
        raise ValueError(
            "Missing exogenous allocation ratios after forward fill for required years."
        )
    _validate_ratio_sums(
        exog.loc[:, ["st", "country"] + ordered_years],
        ["country"],
        ordered_years,
        "exog_st_country_annual_increment_allocation_ratio.csv",
    )
    exog[ordered_years] = exog[ordered_years].div(
        exog.groupby("country")[ordered_years].transform("sum")
    )
    return exog


def _read_fixed_allocation(datacenter_dir, states):
    path = os.path.join(datacenter_dir, "st_country_allocation_ratio.csv")
    fixed = pd.read_csv(path)
    required_cols = {"st", "country", "ratio"}
    if not required_cols.issubset(fixed.columns):
        raise ValueError(
            "st_country_allocation_ratio.csv must contain st, country, ratio."
        )
    fixed = fixed.loc[fixed["st"].isin(states)].copy()
    fixed["ratio"] = fixed["ratio"].astype(float)
    missing_states = sorted(set(states) - set(fixed["st"]))
    if missing_states:
        raise ValueError(
            "Missing fixed allocation data for states/provinces: " f"{missing_states}"
        )
    _validate_ratio_sums(
        fixed,
        ["country"],
        ["ratio"],
        "st_country_allocation_ratio.csv",
    )
    fixed["ratio"] = fixed["ratio"] / fixed.groupby("country")["ratio"].transform("sum")
    return fixed


def _build_state_increments(
    country_increment,
    exog_allocation,
    fixed_allocation,
    state_country,
    country_years,
    layout_year,
    deployment_type,
):
    all_years = pd.DataFrame({"year": country_years})
    state_frame = state_country.drop_duplicates()
    state_years = state_frame.merge(all_years, how="cross")

    country_long = country_increment.melt(
        id_vars=["scenario", "country"],
        var_name="year",
        value_name="it_load_gw",
    )
    country_long["year"] = country_long["year"].astype(int)

    exog_long = exog_allocation.melt(
        id_vars=["st", "country"], var_name="year", value_name="ratio"
    )
    exog_long["year"] = exog_long["year"].astype(int)

    if deployment_type == 0:
        fixed_years = state_years.loc[
            state_years["year"] > layout_year, ["st", "country", "year"]
        ]
        fixed_long = fixed_years.merge(
            fixed_allocation, on=["st", "country"], how="left"
        )
        if fixed_long["ratio"].isnull().any():
            missing = (
                fixed_long.loc[fixed_long["ratio"].isnull(), ["st", "country"]]
                .drop_duplicates()
                .to_dict("records")
            )
            raise ValueError(
                "Missing fixed state/province allocation ratios for years after the "
                "layout cutoff: "
                f"{missing}"
            )
        allocation_long = pd.concat(
            [
                exog_long.loc[
                    exog_long["year"] <= layout_year, ["st", "country", "year", "ratio"]
                ],
                fixed_long.loc[:, ["st", "country", "year", "ratio"]],
            ],
            ignore_index=True,
        )
    else:
        allocation_long = exog_long.loc[:, ["st", "country", "year", "ratio"]].copy()

    state_increment = state_years.merge(
        country_long.loc[:, ["country", "year", "it_load_gw"]],
        on=["country", "year"],
        how="left",
    ).merge(allocation_long, on=["st", "country", "year"], how="left")
    state_increment["it_load_gw"] = state_increment["it_load_gw"].fillna(0.0)
    state_increment["ratio"] = state_increment["ratio"].fillna(0.0)
    state_increment["it_load_gw"] = (
        state_increment["it_load_gw"] * state_increment["ratio"]
    )
    state_increment["capacity_gw"] = state_increment["it_load_gw"]
    return state_increment.loc[
        :, ["st", "country", "year", "capacity_gw", "it_load_gw"]
    ].rename(columns={"st": "region"})


def _make_state_fsa_water_cost(reeds_path, inputs_case, hierarchy, region_col, states):
    cost = pd.read_csv(
        os.path.join(reeds_path, "inputs", "waterclimate", "wat_access_cap_cost.csv")
    )
    required_cols = {"*wst", "sc_cat", "r", "value"}
    if not required_cols.issubset(cost.columns):
        raise ValueError(
            "wat_access_cap_cost.csv must contain columns: *wst, sc_cat, r, value."
        )
    cost = cost.loc[
        (cost["*wst"] == "fsa") & (cost["sc_cat"] == "cost"), ["r", "value"]
    ].copy()
    cost["r"] = cost["r"].astype(str)

    state_map = (
        hierarchy[[region_col, "st"]]
        .drop_duplicates()
        .rename(columns={region_col: "r"})
        .copy()
    )
    state_map["r"] = state_map["r"].astype(str)

    r_ba_path = os.path.join(inputs_case, "r_ba.csv")
    if os.path.exists(r_ba_path):
        ba_map = pd.read_csv(r_ba_path)
        if not {"r", "ba"}.issubset(ba_map.columns):
            raise ValueError("r_ba.csv must contain r and ba columns.")
        ba_map["r"] = ba_map["r"].astype(str)
        ba_map["ba"] = ba_map["ba"].astype(str)
        cost = cost.rename(columns={"r": "ba"})
        state_cost = (
            ba_map.merge(state_map, on="r", how="left")
            .merge(cost, on="ba", how="left")
            .groupby("st", as_index=False)["value"]
            .mean()
        )
    else:
        state_cost = (
            state_map.merge(cost, on="r", how="left")
            .groupby("st", as_index=False)["value"]
            .mean()
        )

    state_cost = state_cost.set_index("st")["value"].reindex(states)
    missing = sorted(state_cost[state_cost.isnull()].index.tolist())
    if missing:
        raise ValueError(
            "Missing average fsa water access cost data for states/provinces: "
            f"{missing}"
        )
    return state_cost.rename("cost_per_mgal_per_year").rename_axis("*st").reset_index()


def _write_flex_timeseries_outputs(inputs_case, inference_multiplier, train_daily):
    """Write the inputs that must track the currently available stress periods."""
    inference_multiplier.round(9).to_csv(
        os.path.join(inputs_case, "datacenter_infer_multiplier.csv"), index=False
    )
    train_daily["days"].to_csv(
        os.path.join(inputs_case, "datacenter_train_day.csv"), index=False, header=False
    )
    train_daily["day_hours"].to_csv(
        os.path.join(inputs_case, "datacenter_train_day_h.csv"),
        index=False,
        header=False,
    )
    train_daily["weights"].rename(columns={"day": "*day"}).round(9).to_csv(
        os.path.join(inputs_case, "datacenter_train_day_weight.csv"), index=False
    )
    train_daily["shares"].round(9).to_csv(
        os.path.join(inputs_case, "datacenter_train_intraday_share.csv"), index=False
    )
    train_daily["routes"].to_csv(
        os.path.join(inputs_case, "datacenter_train_route.csv"),
        index=False,
        header=False,
    )
    train_daily["route_source"].to_csv(
        os.path.join(inputs_case, "datacenter_train_route_source.csv"),
        index=False,
        header=False,
    )
    train_daily["route_destination"].to_csv(
        os.path.join(inputs_case, "datacenter_train_route_destination.csv"),
        index=False,
        header=False,
    )
    train_daily["route_delay"].round(9).to_csv(
        os.path.join(inputs_case, "datacenter_train_route_delay.csv"),
        index=False,
        header=False,
    )
    for key, filename in {
        "route_rep_wrap": "datacenter_train_route_rep_wrap.csv",
        "route_stress_cross_day": "datacenter_train_route_stress_cross_day.csv",
        "route_stress_wrap": "datacenter_train_route_stress_wrap.csv",
    }.items():
        train_daily[key].to_csv(
            os.path.join(inputs_case, filename), index=False, header=False
        )


def refresh_flex_timeseries_inputs(reeds_path, inputs_case):
    """Refresh data-center inputs after Augur creates a new stress iteration.

    GAMS reloads these CSV files before the next solve.  Rebuilding them here
    ensures newly selected stress hours have rigid inference demand and complete
    daily spatial conservation and cyclic temporal-routing mappings.
    """
    sw = reeds.io.get_switches(inputs_case)
    datacenter_dir = os.path.join(reeds_path, "inputs", "datacenter")
    hierarchy = pd.read_csv(os.path.join(inputs_case, "hierarchy.csv"))
    region_col = _get_region_column(hierarchy)
    hierarchy[region_col] = hierarchy[region_col].astype(str)
    hierarchy = hierarchy[[region_col, "st", "country"]].drop_duplicates()
    states = sorted(hierarchy["st"].dropna().unique())
    region_timezones = _make_region_timezones(
        datacenter_dir, inputs_case, hierarchy, region_col, states
    )
    inference_multiplier = _make_inference_multiplier(
        inputs_case, region_timezones, _read_inference_trace(datacenter_dir)
    )
    train_daily = _make_train_daily_inputs(
        inputs_case, sw, hierarchy[region_col].astype(str).unique()
    )
    _write_flex_timeseries_outputs(inputs_case, inference_multiplier, train_daily)


def main(reeds_path, inputs_case):
    sw = reeds.io.get_switches(inputs_case)
    scenario = _normalize_scenario(sw.get("Sw_DatacenterScenario", "mid"))
    layout_year = int(sw.get("Sw_DatacenterLayoutYear", 2028))
    deployment_type = _validate_deployment_type(
        sw.get("Sw_DatacenterDeploymentType", 0)
    )
    endogenous_share = float(sw.get("Sw_DatacenterEndogenousShare", 0.5))
    train_temporal_share = float(sw.get("Sw_DatacenterTrainTemporalFlexShare", 0.4))
    train_max_delay_hours = float(sw.get("Sw_DatacenterTrainMaxDelayHours", 3))
    spatial_scope_raw = float(sw.get("Sw_DatacenterSpatialFlexScope", 0))
    spatial_scope = int(spatial_scope_raw)
    spatial_share = float(sw.get("Sw_DatacenterSpatialFlexShare", 0.4))
    move_penalty = float(sw.get("Sw_DatacenterMovePenalty", 0.001))
    if not 0.0 <= endogenous_share <= 1.0:
        raise ValueError("Sw_DatacenterEndogenousShare must be between 0 and 1.")
    _validate_flex_switches(
        train_temporal_share,
        train_max_delay_hours,
        spatial_scope,
        spatial_share,
        move_penalty,
    )
    if spatial_scope_raw != spatial_scope:
        raise ValueError("Sw_DatacenterSpatialFlexScope must be one of 0, 1, or 2.")

    datacenter_dir = os.path.join(reeds_path, "inputs", "datacenter")

    hierarchy = pd.read_csv(os.path.join(inputs_case, "hierarchy.csv"))
    region_col = _get_region_column(hierarchy)
    hierarchy[region_col] = hierarchy[region_col].astype(str)
    hierarchy = hierarchy[[region_col, "st", "country"]].drop_duplicates()
    states = sorted(hierarchy["st"].dropna().unique())
    years = _read_modeled_years(inputs_case)
    state_country = hierarchy[["st", "country"]].drop_duplicates()

    country_increment, increment_year_cols = _read_country_increments(
        datacenter_dir, scenario
    )
    country_years = sorted(int(year) for year in increment_year_cols)
    max_required_year = max(max(years), max(country_years))
    exog_allocation = _read_exogenous_allocation(
        datacenter_dir, states, max_required_year
    )
    fixed_allocation = _read_fixed_allocation(datacenter_dir, states)
    incremental = _build_state_increments(
        country_increment=country_increment,
        exog_allocation=exog_allocation,
        fixed_allocation=fixed_allocation,
        state_country=state_country,
        country_years=country_years,
        layout_year=layout_year,
        deployment_type=deployment_type,
    )
    incremental["year"] = incremental["year"].astype(int)

    pue = _read_state_yearly_metric(
        datacenter_dir, "st_yearly_pue.csv", "pue", states, years
    )
    train_infer_utilization = _read_train_infer_utilization(datacenter_dir, years)
    inference_trace = _read_inference_trace(datacenter_dir)
    wue = _read_state_yearly_metric(
        datacenter_dir, "st_yearly_wue.csv", "wue_L_per_kWh", states, years
    )
    water_cost = _make_state_fsa_water_cost(
        reeds_path=reeds_path,
        inputs_case=inputs_case,
        hierarchy=hierarchy,
        region_col=region_col,
        states=states,
    )

    ratios = _make_region_state_ratios(
        datacenter_dir=datacenter_dir,
        inputs_case=inputs_case,
        hierarchy=hierarchy,
        region_col=region_col,
        states=states,
    )
    if ratios.empty:
        raise ValueError("No BA-to-state data center load ratios match this case.")
    supply_curve = _make_region_supply_curve(
        reeds_path=reeds_path,
        inputs_case=inputs_case,
        hierarchy=hierarchy,
        region_col=region_col,
    )
    supply_curve = _append_supply_curve_overflow_bin(supply_curve, inputs_case)
    region_timezones = _make_region_timezones(
        datacenter_dir, inputs_case, hierarchy, region_col, states
    )
    inference_multiplier = _make_inference_multiplier(
        inputs_case, region_timezones, inference_trace
    )
    spatial_routes, move_distance = _make_spatial_flex_inputs(
        reeds_path, inputs_case, hierarchy, region_col, spatial_scope
    )
    train_daily = _make_train_daily_inputs(
        inputs_case, sw, hierarchy[region_col].astype(str).unique()
    )

    cumulative = _make_cumulative_it(incremental, states, years)
    floor = _make_incremental_floor(incremental, states, years, layout_year)
    exogenous_supply_curve_requirement = _make_exogenous_supply_curve_requirement(
        cumulative=cumulative,
        floor=floor,
        ratios=ratios,
        supply_curve=supply_curve,
        layout_year=layout_year,
        deployment_type=deployment_type,
        endogenous_share=endogenous_share,
    )

    country_increment_long = country_increment.melt(
        id_vars=["scenario", "country"], var_name="year", value_name="it_load_gw"
    )
    country_increment_long["year"] = country_increment_long["year"].astype(int)
    cumulative_country = (
        country_increment_long.groupby(["country", "year"], as_index=False)[
            "it_load_gw"
        ]
        .sum()
        .sort_values(["country", "year"])
    )
    cumulative_country["MW"] = (
        cumulative_country.groupby("country")["it_load_gw"].cumsum() * 1000.0
    )
    cumulative_country = cumulative_country.loc[
        cumulative_country["year"].isin(years), ["country", "year", "MW"]
    ].rename(columns={"country": "*country", "year": "t"})
    cumulative_total = (
        cumulative_country.groupby("t", as_index=False)["MW"]
        .sum()
        .rename(columns={"t": "*t"})
    )

    pue.round(6).to_csv(os.path.join(inputs_case, "datacenter_pue.csv"), index=False)
    train_infer_utilization.round(9).to_csv(
        os.path.join(inputs_case, "datacenter_train_infer_util.csv"), index=False
    )
    _write_flex_timeseries_outputs(inputs_case, inference_multiplier, train_daily)
    spatial_routes.to_csv(
        os.path.join(inputs_case, "datacenter_spatial_route.csv"),
        index=False,
        header=False,
    )
    move_distance.round(9).to_csv(
        os.path.join(inputs_case, "datacenter_move_distance.csv"), index=False
    )
    wue.round(6).to_csv(os.path.join(inputs_case, "datacenter_wue.csv"), index=False)
    water_cost.round(6).to_csv(
        os.path.join(inputs_case, "datacenter_water_cost.csv"), index=False
    )
    ratios.round(9).to_csv(
        os.path.join(inputs_case, "datacenter_r_st_ratio.csv"), index=False
    )
    cumulative.rename(columns={"st": "*st"}).round(6).to_csv(
        os.path.join(inputs_case, "datacenter_it_exog.csv"), index=False
    )
    floor.rename(columns={"st": "*st"}).round(6).to_csv(
        os.path.join(inputs_case, "datacenter_it_min.csv"), index=False
    )
    cumulative_country.round(6).to_csv(
        os.path.join(inputs_case, "datacenter_country_it_exog.csv"), index=False
    )
    cumulative_total.round(6).to_csv(
        os.path.join(inputs_case, "datacenter_total_it_exog.csv"), index=False
    )
    supply_curve.round(6).to_csv(
        os.path.join(inputs_case, "datacenter_supply_curve.csv"), index=False
    )
    exogenous_supply_curve_requirement.round(6).to_csv(
        os.path.join(inputs_case, "datacenter_sc_exog_req.csv"), index=False
    )


if __name__ == "__main__" and not hasattr(sys, "ps1"):
    parser = argparse.ArgumentParser(description="Prepare data center inputs")
    parser.add_argument("reeds_path", help="ReEDS directory")
    parser.add_argument("inputs_case", help="output directory")
    args = parser.parse_args()

    tic = datetime.datetime.now()
    log = reeds.log.makelog(
        scriptname=__file__,
        logpath=os.path.join(args.inputs_case, "..", "gamslog.txt"),
    )

    print("Starting datacenter.py")
    main(args.reeds_path, args.inputs_case)
    reeds.log.toc(
        tic=tic,
        year=0,
        process="input_processing/datacenter.py",
        path=os.path.join(args.inputs_case, ".."),
    )
    print("Finished datacenter.py")
