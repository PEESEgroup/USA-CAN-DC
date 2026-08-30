from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import sys
import tempfile
import time

import click
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from .calculate_gravity_score import calc_gravity_array_from_distance, calc_gravity_score
from .calculate_locational_cost import calculate_locational_cost
from .configure_output import configure_output
from .determine_sites import build_graph, get_region_suit_array, site_based_on_siting_score
from .load_data import (
    collect_constraints,
    find_region_window,
    get_yaml,
    load_raster_array,
    load_region_raster,
    validate_raster_alignment,
)
from .postprocess_output import generate_output_artifacts
from .utils import convert_sqft_to_grid_cells, get_normalized_value


def _get_binary_setting(settings: dict, name: str, default: int = 1) -> bool:
    """Return a YAML 0/1 setting as a boolean."""
    value = settings.get(name, default)
    if value not in (0, 1, False, True):
        raise ValueError(f"settings.{name} must be 0 or 1; received {value!r}.")
    return bool(value)


def _get_interconnection_distance_km(
    raw_distance: float,
    settings: dict,
    transform,
) -> float:
    """Convert the configured interconnection-distance raster value to km."""
    unit = settings.get('interconnection_distance_unit', 'km')
    if unit == 'km':
        return raw_distance
    if unit == 'grid_cells':
        cell_width_m = abs(transform.a)
        cell_height_m = abs(transform.e)
        if cell_width_m != cell_height_m:
            raise ValueError(
                'A grid-cell interconnection distance requires square raster cells; '
                f'found {cell_width_m} m by {cell_height_m} m.'
            )
        return raw_distance * cell_width_m / 1_000
    raise ValueError(
        "settings.interconnection_distance_unit must be 'km' or 'grid_cells'; "
        f"received {unit!r}."
    )


def _resolve_region_scenarios(
    region_name: str,
    region_dict: dict,
    global_scenarios: dict | None,
) -> list[tuple[str, dict]]:
    """
    Return the scenario configs to run for a region.

    Supports:
    - legacy single-scenario region configs
    - a top-level `scenarios` map shared by all regions
    - per-region `scenarios` overrides
    """
    if 'scenarios' in region_dict:
        scenario_dict = region_dict['scenarios']
    else:
        scenario_dict = global_scenarios

    if not scenario_dict:
        return [('default', region_dict)]

    base_region_dict = {k: v for k, v in region_dict.items() if k != 'scenarios'}
    resolved_scenarios: list[tuple[str, dict]] = []
    for scenario_name, scenario_config in scenario_dict.items():
        merged_config = {**base_region_dict, **scenario_config}
        merged_config['scenario_name'] = scenario_name
        resolved_scenarios.append((scenario_name, merged_config))

    if not resolved_scenarios:
        raise ValueError(f"Region {region_name} does not define any scenarios to run.")

    return resolved_scenarios


def _drop_overlapping_smaller_sites(output_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Remove smaller sites whose polygons overlap with already-kept larger sites.
    """
    if output_gdf.empty:
        return output_gdf

    sort_columns = ['campus_size_square_ft', 'data_center_it_power_mw', 'total_capex_cost']
    available_columns = [column for column in sort_columns if column in output_gdf.columns]
    ranked_gdf = output_gdf.sort_values(
        by=available_columns,
        ascending=[False] * len(available_columns),
        kind='mergesort',
    ).reset_index(drop=True)

    kept_indices: list[int] = []
    kept_geometries = []
    for idx, row in ranked_gdf.iterrows():
        geometry = row.geometry
        overlaps_larger_site = any(
            geometry.intersects(kept_geometry) and geometry.intersection(kept_geometry).area > 0
            for kept_geometry in kept_geometries
        )
        if overlaps_larger_site:
            continue
        kept_indices.append(idx)
        kept_geometries.append(geometry)

    return ranked_gdf.loc[kept_indices].reset_index(drop=True)


def _get_selected_region_names(
    expansion_dict: dict[str, dict],
    selected_region_ids: list[int],
) -> list[str]:
    selected_ids = set(selected_region_ids)
    region_names = [
        region_name
        for region_name, region_config in expansion_dict.items()
        if region_config['region_id'] in selected_ids
    ]
    if not region_names:
        raise ValueError("No regions from selected_region_ids are present in expansion_plan.")
    return region_names


def _prepare_runtime_context(
    config_dict: dict,
    selected_region_ids: list[int],
    logger: logging.Logger,
) -> dict:
    settings_dict = config_dict['settings']
    constraint_dict = config_dict['constraints']
    market_dict = config_dict['market_gravity']
    interconnection_only_siting = _get_binary_setting(
        settings_dict, 'interconnection_only_siting', default=0
    )

    if interconnection_only_siting:
        raster_names = ['interconnection_distance_km']
        raster_paths = [constraint_dict['interconnection_distance_km']]
    else:
        raster_names = list(constraint_dict.keys())
        raster_paths = list(constraint_dict.values())
    all_raster_paths = [
        settings_dict['region_raster_path'],
        settings_dict['siting_raster_path'],
        *raster_paths,
    ]
    if not interconnection_only_siting and settings_dict.get('market_weight', 0.0):
        all_raster_paths.append(market_dict['market_raster_path'])

    logger.info("Validating raster alignment...")
    validate_raster_alignment(settings_dict['region_raster_path'], all_raster_paths[1:])

    logger.info("Computing minimum raster window for selected regions: %s", selected_region_ids)
    region_window = find_region_window(settings_dict['region_raster_path'], selected_region_ids)
    logger.info(
        "Using raster window row_off=%s col_off=%s height=%s width=%s",
        int(region_window.row_off),
        int(region_window.col_off),
        int(region_window.height),
        int(region_window.width),
    )

    logger.info("Loading region raster from %s", settings_dict['region_raster_path'])
    region_array, transform = load_region_raster(settings_dict['region_raster_path'], window=region_window)

    logger.info("Loading siting suitability raster from %s", settings_dict['siting_raster_path'])
    suit_array = load_raster_array(raster_fn=settings_dict['siting_raster_path'], window=region_window)
    batch_region_mask = np.isin(region_array, selected_region_ids)
    batch_suit_array = np.where(batch_region_mask, suit_array, 0)

    if interconnection_only_siting:
        distance_array = load_raster_array(raster_paths[0], window=region_window)
        distance_km = _get_interconnection_distance_km(distance_array, settings_dict, transform)
        batch_suit_array = np.where(distance_km <= 2, batch_suit_array, 0)

    node_values = collect_constraints(
        batch_suit_array,
        transform,
        raster_paths,
        raster_names,
        logger,
        window=region_window,
    )

    market_weight = 0.0 if interconnection_only_siting else settings_dict.get('market_weight', 0.0)
    if market_weight:
        market_array = load_raster_array(
            raster_fn=market_dict['market_raster_path'], window=region_window
        )
        logger.info("Calculating market gravity multipliers...")
        gravity_multiplier_array = calc_gravity_array_from_distance(market_array, batch_suit_array)
    else:
        logger.info("Skipping market gravity calculation because market_weight is 0.")
        gravity_multiplier_array = None

    return {
        'settings_dict': settings_dict,
        'expansion_dict': config_dict['expansion_plan'],
        'global_scenarios': config_dict.get('scenarios'),
        'raster_names': raster_names,
        'region_array': region_array,
        'transform': transform,
        'batch_suit_array': batch_suit_array,
        'node_values': node_values,
        'gravity_multiplier_array': gravity_multiplier_array,
        'market_weight': market_weight,
        'interconnection_only_siting': interconnection_only_siting,
    }


def _run_region_in_context(
    region_name: str,
    region_dict: dict,
    runtime_context: dict,
    logger: logging.Logger,
) -> gpd.GeoDataFrame:
    settings_dict = runtime_context['settings_dict']
    global_scenarios = runtime_context['global_scenarios']
    raster_names = runtime_context['raster_names']
    region_array = runtime_context['region_array']
    transform = runtime_context['transform']
    batch_suit_array = runtime_context['batch_suit_array']
    node_values = runtime_context['node_values']
    gravity_multiplier_array = runtime_context['gravity_multiplier_array']
    market_weight = runtime_context['market_weight']
    interconnection_only_siting = runtime_context['interconnection_only_siting']
    enable_land_and_real_property_tax = _get_binary_setting(
        settings_dict, 'enable_land_and_real_property_tax'
    )
    enable_personal_property_tax = _get_binary_setting(
        settings_dict, 'enable_personal_property_tax'
    )
    enable_sales_tax = _get_binary_setting(settings_dict, 'enable_sales_tax')
    enable_industrial_electricity_and_cooling = _get_binary_setting(
        settings_dict, 'enable_industrial_electricity_and_cooling'
    )

    logger.info("Processing region: %s", region_name)

    region_scenarios = _resolve_region_scenarios(region_name, region_dict, global_scenarios)
    if interconnection_only_siting:
        region_scenarios.sort(key=lambda item: item[1]['campus_size_square_ft'], reverse=True)
    region_id = region_dict['region_id']
    region_suit_array = get_region_suit_array(region_array, batch_suit_array, region_id)

    region_frames: list[gpd.GeoDataFrame] = []
    for scenario_name, scenario_dict in region_scenarios:
        logger.info("Processing scenario %s for region %s", scenario_name, region_name)

        campus_size_square_ft = scenario_dict['campus_size_square_ft']
        cooling_water_intensity_gal_per_mwh = scenario_dict.get('cooling_water_intensity_gal_per_mwh', 460)
        cooling_water_consumption_fraction = scenario_dict.get('cooling_water_consumption_fraction', .8)
        facility_overhead_frac = scenario_dict.get('facility_overhead_frac', 0)
        equipment_capex_usd = scenario_dict['equipment_capital_expenditure_usd']
        building_capex_usd = scenario_dict['building_capital_expenditure_usd']
        assessed_real_property_frac = scenario_dict.get('assessed_real_property_frac', .16)
        assessed_personal_property_frac = scenario_dict.get('assessed_personal_property_frac', .8)
        data_center_it_power_mw = scenario_dict['data_center_it_power_mw']
        data_center_pue = scenario_dict['data_center_pue']
        interconnection_cost_usd_per_km = scenario_dict['interconnection_cost_usd_per_km']
        number_of_sites = None if interconnection_only_siting else scenario_dict['n_sites']
        min_block_size = int(convert_sqft_to_grid_cells(campus_size_square_ft))

        logger.info("Building siting graph for scenario %s...", scenario_name)
        G = build_graph(region_suit_array, min_block_size, raster_names, node_values)
        if G.number_of_nodes() == 0:
            logger.warning(
                "No suitable connected components met the minimum block size for region %s scenario %s.",
                region_name,
                scenario_name,
            )
            continue

        if interconnection_only_siting:
            for node, attrs in G.nodes(data=True):
                distance_km = _get_interconnection_distance_km(
                    attrs['interconnection_distance_km'], settings_dict, transform
                )
                G.nodes[node]['parameters'] = {
                    'campus_size_square_ft': campus_size_square_ft,
                    'it_power_mw': data_center_it_power_mw,
                    'interconnection_cost_usd': (
                        distance_km * interconnection_cost_usd_per_km
                    ),
                }

            result_list = site_based_on_siting_score(
                G,
                number_of_sites,
                min_block_size,
                region_name,
                transform,
                attribute=None,
            )
            for result in result_list:
                selected_nodes = result[next(iter(result))]['row_col_list']
                rows, cols = zip(*selected_nodes)
                region_suit_array[np.asarray(rows), np.asarray(cols)] = 0

            scenario_gdf = configure_output(
                result_list,
                region_id,
                output_prefix=f"{region_id}_{scenario_name}",
                extra_fields={'scenario': scenario_name},
                interconnection_only=True,
            )
            region_frames.append(scenario_gdf)
            continue

        logger.info("Calculating locational costs for scenario %s...", scenario_name)
        for node, attrs in tqdm(list(G.nodes(data=True))):
            G.nodes[node]['locational_cost'], node_parameter_dict = calculate_locational_cost(
                campus_size_square_ft=campus_size_square_ft,
                land_cost_usd_per_sqft=attrs['land_cost_per_sqft'],
                elec_rate_usd_per_kwh=attrs['electricity_rate_per_kwh'],
                personal_prop_tax_rate=attrs['personal_prop_tax_rate'],
                real_property_tax_rate=attrs['real_property_tax_rate'],
                sales_tax_rate=attrs['sales_tax_rate'],
                interconnection_distance_km=_get_interconnection_distance_km(
                    attrs['interconnection_distance_km'], settings_dict, transform
                ),
                mechanical_cool_fraction=attrs['mechanical_cool_fraction'],
                water_cool_fraction=attrs['water_cool_fraction'],
                equipment_capex_usd=equipment_capex_usd,
                building_capex_usd=building_capex_usd,
                interconnection_cost_usd_per_km=interconnection_cost_usd_per_km,
                data_center_it_power_mw=data_center_it_power_mw,
                data_center_pue=data_center_pue,
                assessed_real_property_frac=assessed_real_property_frac,
                assessed_personal_property_frac=assessed_personal_property_frac,
                cooling_water_intensity_gal_per_mwh=cooling_water_intensity_gal_per_mwh,
                cooling_water_consumption_fraction=cooling_water_consumption_fraction,
                facility_overhead_frac=facility_overhead_frac,
                enable_land_and_real_property_tax=enable_land_and_real_property_tax,
                enable_personal_property_tax=enable_personal_property_tax,
                enable_sales_tax=enable_sales_tax,
                enable_industrial_electricity_and_cooling=enable_industrial_electricity_and_cooling,
            )
            G.nodes[node]['parameters'] = node_parameter_dict

        max_locational_cost = max(data['locational_cost'] for _, data in G.nodes(data=True))
        min_locational_cost = min(data['locational_cost'] for _, data in G.nodes(data=True))

        if market_weight:
            logger.info("Calculating market gravity score for scenario %s...", scenario_name)
            for node, _attrs in tqdm(list(G.nodes(data=True))):
                G.nodes[node]['gravity_score'] = calc_gravity_score(
                    node,
                    gravity_multiplier_array,
                    data_center_it_power_mw,
                    alpha=0.5,
                )

            max_gravity_score = max(data['gravity_score'] for _, data in G.nodes(data=True))
            min_gravity_score = min(data['gravity_score'] for _, data in G.nodes(data=True))

        logger.info("Normalizing cost and market score for scenario %s...", scenario_name)
        cost_weight = settings_dict.get('cost_weight', 1.0)
        for node, _attrs in tqdm(list(G.nodes(data=True))):
            G.nodes[node]['normalized_locational_cost'] = get_normalized_value(
                G,
                attribute='locational_cost',
                node=node,
                max_value=max_locational_cost,
                min_value=min_locational_cost,
            )
            if market_weight:
                G.nodes[node]['normalized_gravity_score'] = get_normalized_value(
                    G,
                    attribute='gravity_score',
                    node=node,
                    max_value=max_gravity_score,
                    min_value=min_gravity_score,
                )
            else:
                G.nodes[node]['normalized_gravity_score'] = 0

            norm_loc_cost = G.nodes[node]['normalized_locational_cost']
            norm_gravity_score = G.nodes[node]['normalized_gravity_score']
            G.nodes[node]['total_weighted_siting_score'] = (
                cost_weight * norm_loc_cost
            ) + (
                market_weight * norm_gravity_score
            )

        result_list = site_based_on_siting_score(
            G,
            number_of_sites,
            min_block_size,
            region_name,
            transform,
        )

        logger.info(
            "Sited %s data center(s) in region %s for scenario %s.",
            len(result_list),
            region_name,
            scenario_name,
        )

        scenario_gdf = configure_output(
            result_list,
            region_id,
            output_prefix=f"{region_id}_{scenario_name}",
            extra_fields={'scenario': scenario_name},
        )
        region_frames.append(scenario_gdf)

    return _concatenate_output_frames(region_frames)


def _concatenate_output_frames(frames: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs='ESRI:102003')
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry='geometry', crs='ESRI:102003')


def _sort_output_gdf(output_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if output_gdf.empty:
        return output_gdf

    sort_columns = [column for column in ['region', 'scenario', 'id'] if column in output_gdf.columns]
    if not sort_columns:
        return output_gdf.reset_index(drop=True)
    return output_gdf.sort_values(by=sort_columns, kind='mergesort').reset_index(drop=True)


def _write_geojson_file(output_gdf: gpd.GeoDataFrame, output_file: str | Path) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    web_gdf = output_gdf if output_gdf.crs == 'EPSG:4326' else output_gdf.to_crs(epsg=4326)
    path.write_text(web_gdf.to_json(), encoding='utf-8')


def _finalize_output_gdf(output_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if 'geometry' in output_gdf.columns:
        output_gdf = _drop_overlapping_smaller_sites(
            gpd.GeoDataFrame(output_gdf, geometry='geometry', crs='ESRI:102003')
        )
    return _sort_output_gdf(output_gdf)


def _save_output(
    output_gdf: gpd.GeoDataFrame,
    output_file: str | Path | None,
    logger: logging.Logger,
    include_artifacts: bool = True,
) -> None:
    if not output_file:
        return

    logger.info("Saving output to %s", output_file)
    _write_geojson_file(output_gdf, output_file)
    if include_artifacts:
        html_path, csv_path = generate_output_artifacts(output_file)
        logger.info("Saved HTML map to %s", html_path)
        logger.info("Saved CSV to %s", csv_path)


def _run_serial_from_config_dict(
    config_dict: dict,
    selected_region_names: list[str],
    logger: logging.Logger,
) -> gpd.GeoDataFrame:
    selected_region_ids = [config_dict['expansion_plan'][name]['region_id'] for name in selected_region_names]
    runtime_context = _prepare_runtime_context(config_dict, selected_region_ids, logger)

    logger.info("Starting siting process...")
    region_frames = [
        _run_region_in_context(region_name, config_dict['expansion_plan'][region_name], runtime_context, logger)
        for region_name in selected_region_names
    ]
    return _concatenate_output_frames(region_frames)


def _build_region_subconfig(config_dict: dict, region_name: str, output_file: str) -> dict:
    region_dict = config_dict['expansion_plan'][region_name]
    region_id = region_dict['region_id']
    subconfig = json.loads(json.dumps(config_dict))
    subconfig['expansion_plan'] = {region_name: region_dict}
    subconfig.setdefault('settings', {})
    subconfig['settings']['selected_region_ids'] = [region_id]
    subconfig['settings']['output_file'] = str(Path(output_file).resolve()).replace("\\", "/")
    return subconfig


def _run_region_worker(config_path: str, region_name: str, output_path: str) -> str:
    fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt)
    logger = logging.getLogger(__name__)

    config_dict = get_yaml(config_path)
    subconfig = _build_region_subconfig(config_dict, region_name, output_path)
    output_gdf = _run_serial_from_config_dict(subconfig, [region_name], logger)
    output_gdf = _finalize_output_gdf(output_gdf)
    _save_output(output_gdf, output_path, logger, include_artifacts=False)
    return output_path


def _run_regions_in_parallel(
    config_path: str,
    config_dict: dict,
    selected_region_names: list[str],
    max_workers: int,
    logger: logging.Logger,
) -> gpd.GeoDataFrame:
    with tempfile.TemporaryDirectory(prefix='cerf_region_runs_') as temp_dir:
        future_to_region = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for index, region_name in enumerate(selected_region_names):
                output_path = Path(temp_dir) / f"region_{index:03d}_{region_name}.geojson"
                future = executor.submit(_run_region_worker, config_path, region_name, str(output_path))
                future_to_region[future] = region_name

            completed_outputs: dict[str, Path] = {}
            for future in as_completed(future_to_region):
                region_name = future_to_region[future]
                try:
                    output_path = future.result()
                except Exception as exc:
                    for pending_future in future_to_region:
                        pending_future.cancel()
                    raise RuntimeError(f"Parallel siting failed for region {region_name}") from exc
                completed_outputs[region_name] = Path(output_path)
                logger.info("Finished parallel region: %s", region_name)

        region_frames = [
            gpd.read_file(completed_outputs[region_name]).to_crs('ESRI:102003')
            for region_name in selected_region_names
            if region_name in completed_outputs
        ]
        return _concatenate_output_frames(region_frames)


def run(config: str, max_workers: int | None = None) -> gpd.GeoDataFrame:
    """
    Run the data center siting process based on the provided configuration file.
    """
    t0 = time.time()
    fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt)
    logger = logging.getLogger(__name__)

    logger.info("Application started")
    logger.info("Using configuration file: %s", config)
    config_dict = get_yaml(config)

    expansion_dict = config_dict['expansion_plan']
    settings_dict = config_dict['settings']
    selected_region_ids = settings_dict.get(
        'selected_region_ids',
        [region_cfg['region_id'] for region_cfg in expansion_dict.values()]
    )
    selected_region_names = _get_selected_region_names(expansion_dict, selected_region_ids)

    worker_count = 1 if max_workers in (None, 0) else max_workers
    if worker_count < 1:
        raise ValueError("max_workers must be at least 1.")

    if worker_count > 1 and len(selected_region_names) > 1:
        output_gdf = _run_regions_in_parallel(config, config_dict, selected_region_names, worker_count, logger)
    else:
        output_gdf = _run_serial_from_config_dict(config_dict, selected_region_names, logger)

    output_gdf = _finalize_output_gdf(output_gdf)
    _save_output(output_gdf, settings_dict.get('output_file'), logger, include_artifacts=True)

    logger.info("All regions processed in %s minutes.", round((time.time() - t0) / 60, 2))
    return output_gdf


@click.group()
def cli():
    """Data Center Siting Tool - Identifies optimal locations for data centers based on various constraints and costs."""
    pass


@cli.command()
@click.argument('config', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('-o', '--output', type=click.Path(dir_okay=False, path_type=Path),
              help='Path to save the output GeoJSON file (overrides config file setting)')
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose logging output')
@click.option('--log-file', type=click.Path(dir_okay=False, path_type=Path),
              help='Path to save the log file')
@click.option('--workers', type=int, default=1, show_default=True,
              help='Number of worker processes to use for region-level parallel execution')
def site(config: Path, output: Path, verbose: bool, log_file: Path, workers: int):
    """Run the data center siting process using the provided configuration file."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    if log_file:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=log_level,
            format=log_format
        )

    logger = logging.getLogger(__name__)

    try:
        output_gdf = run(str(config), max_workers=workers)

        if output:
            logger.info("Saving output to %s", output)
            _write_geojson_file(output_gdf, output)
            html_path, csv_path = generate_output_artifacts(output)
            logger.info("Saved HTML map to %s", html_path)
            logger.info("Saved CSV to %s", csv_path)

    except Exception as e:
        logger.error("Error running data center siting: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    cli()
