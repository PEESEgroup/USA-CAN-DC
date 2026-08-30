from typing import List, Dict, Any

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer


_WEB_COORD_TRANSFORMER = Transformer.from_crs("ESRI:102003", "EPSG:4326", always_xy=True)


def configure_output(
    result_list: List[Dict[int, Dict[str, Any]]],
    region_id: int,
    output_prefix: str | None = None,
    extra_fields: dict[str, Any] | None = None,
    interconnection_only: bool = False,
) -> gpd.GeoDataFrame:
    """
    Configure siting results into a GeoDataFrame with only the requested
    output fields.
    """
    if not result_list:
        if interconnection_only:
            return gpd.GeoDataFrame(
                columns=[
                    'scenario', 'campus_size_square_ft', 'selected_grid_cell_count',
                    'data_center_it_power_mw', 'interconnection cost', 'longitude',
                    'latitude', 'geometry',
                ],
                geometry='geometry', crs='ESRI:102003',
            )
        return gpd.GeoDataFrame(
            columns=[
                'id',
                'region',
                'longitude',
                'latitude',
                'scenario',
                'campus_size_square_ft',
                'selected_grid_cell_count',
                'data_center_it_power_mw',
                'locational_cost',
                'locational_cost_million_usd',
                'land_cost_usd_per_sqft',
                'building_capex_usd',
                'real_property_tax_rate',
                'personal_prop_tax_rate',
                'sales_tax_rate',
                'land_building_cost_usd',
                'equipment_capex_usd',
                'real_property_taxes_usd',
                'personal_property_taxes_usd',
                'equipment_sales_tax_usd',
                'interconnection cost',
                'total_capex_cost',
                'geometry',
            ],
            geometry='geometry',
            crs='ESRI:102003',
        )

    rows = []
    extra_fields = extra_fields or {}

    for i in range(len(result_list)):
        result = result_list[i][i]
        output_id = f"{output_prefix}_{i}" if output_prefix else f"{region_id}_{i}"
        scenario = extra_fields.get('scenario')
        remaining_extra_fields = {k: v for k, v in extra_fields.items() if k != 'scenario'}

        if interconnection_only:
            for x, y in result['coord_list']:
                rows.append({
                    'id': output_id,
                    'projected_xcoord': x,
                    'projected_ycoord': y,
                    'scenario': scenario,
                    'campus_size_square_ft': result['campus_size_square_ft'],
                    'selected_grid_cell_count': result.get('selected_grid_cell_count'),
                    'data_center_it_power_mw': result.get('it_power_mw'),
                    'interconnection cost': result['interconnection_cost_usd'],
                    **remaining_extra_fields,
                })
            continue

        building_capex_usd = result.get('building_capex_usd', result.get('building_capex', 0))
        equipment_capex_usd = result.get('equipment_capex_usd', result.get('equipment_capex', 0))
        land_building_cost_usd = result.get(
            'land_building_cost_usd',
            result.get('property_cost_usd', 0) + building_capex_usd,
        )
        real_property_taxes_usd = result.get(
            'real_property_taxes_usd',
            result.get('total_property_tax_usd', 0),
        )
        personal_property_taxes_usd = result.get('personal_property_taxes_usd', 0)
        equipment_sales_tax_usd = result.get(
            'equipment_sales_tax_usd',
            result.get('total_sales_tax_usd', 0),
        )
        interconnection_cost_usd = result.get('interconnection_cost_usd', 0)
        total_capex_cost = result.get(
            'total_capex_cost',
            land_building_cost_usd +
            equipment_capex_usd +
            real_property_taxes_usd +
            personal_property_taxes_usd +
            equipment_sales_tax_usd +
            interconnection_cost_usd,
        )

        for x, y in result['coord_list']:
            rows.append({
                'id': output_id,
                'region': result['region_name'],
                'projected_xcoord': x,
                'projected_ycoord': y,
                'scenario': scenario,
                'campus_size_square_ft': result['campus_size_square_ft'],
                'selected_grid_cell_count': result.get('selected_grid_cell_count'),
                'data_center_it_power_mw': result.get('it_power_mw'),
                'locational_cost': result['locational_cost'],
                'locational_cost_million_usd': round(result['locational_cost'] / 1_000_000, 4),
                'land_cost_usd_per_sqft': result.get('land_cost_usd_per_sqft'),
                'building_capex_usd': building_capex_usd,
                'real_property_tax_rate': result['real_property_tax_rate'],
                'personal_prop_tax_rate': result['personal_prop_tax_rate'],
                'sales_tax_rate': result['sales_tax_rate'],
                'land_building_cost_usd': land_building_cost_usd,
                'equipment_capex_usd': equipment_capex_usd,
                'real_property_taxes_usd': real_property_taxes_usd,
                'personal_property_taxes_usd': personal_property_taxes_usd,
                'equipment_sales_tax_usd': equipment_sales_tax_usd,
                'interconnection cost': interconnection_cost_usd,
                'total_capex_cost': total_capex_cost,
                **remaining_extra_fields,
            })

    df = pd.DataFrame(rows)
    geometry = [Point(xy) for xy in zip(df['projected_xcoord'], df['projected_ycoord'])]
    gdf = gpd.GeoDataFrame(df, crs='ESRI:102003', geometry=geometry)
    gdf['geometry'] = gdf['geometry'].buffer(50, cap_style=3)
    gdf = gdf.dissolve(by='id', as_index=False)

    representative_points = gdf.geometry.representative_point()
    lon_lat_pairs = [
        _WEB_COORD_TRANSFORMER.transform(point.x, point.y)
        for point in representative_points
    ]
    gdf['longitude'] = [lon for lon, _lat in lon_lat_pairs]
    gdf['latitude'] = [lat for _lon, lat in lon_lat_pairs]
    gdf = gdf.drop(columns=['projected_xcoord', 'projected_ycoord'])
    if interconnection_only:
        gdf = gdf[
            [
                'scenario', 'campus_size_square_ft', 'selected_grid_cell_count',
                'data_center_it_power_mw', 'interconnection cost', 'longitude',
                'latitude', 'geometry',
            ]
        ]
    return gdf
