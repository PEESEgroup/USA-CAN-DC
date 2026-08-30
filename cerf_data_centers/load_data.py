import os
import time
import logging

import numpy as np
import rasterio
import yaml
from rasterio.windows import Window


def read_yaml(yaml_file: str) -> dict:
    """
    Read a YAML file and return its contents as a dictionary.

    Args:
        yaml_file (str): Path to the YAML file to be read.

    Returns:
        dict: Contents of the YAML file parsed into a dictionary.
    """
    with open(yaml_file, 'r') as yml:
        return yaml.load(yml, Loader=yaml.FullLoader)


def get_yaml(config_file: str) -> dict:
    """
    Read and parse a YAML configuration file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed contents of the YAML file as a dictionary.

    Raises:
        AttributeError: If config_file is None.
        FileNotFoundError: If the specified config_file does not exist.
    """
    if config_file is None:
        msg = "Config file must be passed as an argument using:  config_file='<path to config.yml'>"
        raise AttributeError(msg)
    if os.path.isfile(config_file):
        return read_yaml(config_file)
    else:
        msg = f"Config file not found for path:  {config_file}."
        raise FileNotFoundError(msg)
        

def load_region_raster(
    siting_region_fn: str,
    window: Window | None = None
) -> tuple[np.ndarray, rasterio.Affine]:
    """
    Load the region raster from the specified file path.

    Args:
        siting_region_fn (str): Path to the siting region raster file.

    Returns:
        tuple[np.ndarray, rasterio.Affine]: 
            A tuple containing:
                - region_array (np.ndarray): The raster data as a 2D numpy array.
                - transform (rasterio.Affine): The affine transformation for the raster.
    """
    with rasterio.open(siting_region_fn) as src:
        region_array = src.read(1, window=window)
        transform = src.window_transform(window) if window is not None else src.transform
    return region_array, transform


def load_raster_array(raster_fn: str, window: Window | None = None) -> np.ndarray:
    """
    Load the raster from the specified file path.

    Args:
        raster_fn (str): Path to the raster file.

    Returns:
        np.ndarray: A 2D numpy array representing the raster.
    """
    with rasterio.open(raster_fn) as src:
        suit_array = src.read(1, window=window)
    return suit_array


def find_region_window(region_raster_path: str, selected_region_ids: list[int]) -> Window:
    """
    Find the minimum raster window that contains all requested region IDs.

    Args:
        region_raster_path (str): Path to the region raster.
        selected_region_ids (list[int]): Region IDs to include in the window.

    Returns:
        Window: Minimum bounding raster window containing the requested region IDs.

    Raises:
        ValueError: If none of the selected region IDs are found in the region raster.
    """
    selected_ids = np.asarray(selected_region_ids)
    min_row, min_col = None, None
    max_row, max_col = None, None

    with rasterio.open(region_raster_path) as src:
        for _, window in src.block_windows(1):
            block = src.read(1, window=window)
            matches = np.isin(block, selected_ids)
            if not np.any(matches):
                continue

            rows, cols = np.where(matches)
            block_min_row = int(window.row_off + rows.min())
            block_max_row = int(window.row_off + rows.max())
            block_min_col = int(window.col_off + cols.min())
            block_max_col = int(window.col_off + cols.max())

            min_row = block_min_row if min_row is None else min(min_row, block_min_row)
            max_row = block_max_row if max_row is None else max(max_row, block_max_row)
            min_col = block_min_col if min_col is None else min(min_col, block_min_col)
            max_col = block_max_col if max_col is None else max(max_col, block_max_col)

    if min_row is None or min_col is None or max_row is None or max_col is None:
        raise ValueError(f"None of the selected region IDs were found: {selected_region_ids}")

    return Window.from_slices((min_row, max_row + 1), (min_col, max_col + 1))


def validate_raster_alignment(
    reference_raster_path: str,
    other_raster_paths: list[str]
) -> None:
    """
    Validate that all raster inputs share the same grid definition.

    Args:
        reference_raster_path (str): Reference raster path.
        other_raster_paths (list[str]): Other raster paths to compare against.

    Raises:
        ValueError: If any raster differs in CRS, transform, width, or height.
    """
    with rasterio.open(reference_raster_path) as ref:
        reference = (ref.crs, ref.transform, ref.width, ref.height)

    for path in other_raster_paths:
        with rasterio.open(path) as candidate:
            current = (candidate.crs, candidate.transform, candidate.width, candidate.height)
        if current != reference:
            raise ValueError(
                f"Raster alignment mismatch for {path}. All rasters must share CRS, transform, width, and height."
            )


def collect_constraints(
        suit_array: np.ndarray,
        transform: rasterio.Affine,
        raster_paths: list[str],
        raster_names: list[str],
        logger: logging.Logger,
        window: Window | None = None
    ) -> dict[tuple[int, int], dict[str, float]]:
        """
        Collect and extract cost and constraint data for suitable siting locations.

        This function samples raster values for each constraint/cost layer at the locations
        where the suitability array indicates a suitable site (value == 1). The sampled
        values are stored in a dictionary keyed by (row, col) grid cell indices, with each
        value being a dictionary mapping constraint/cost names to their sampled values.

        Args:
            suit_array (np.ndarray): 2D array indicating suitable siting locations (1 = suitable).
            transform (rasterio.Affine): Affine transformation for converting array indices to coordinates.
            raster_paths (list[str]): List of file paths to cost/constraint raster files.
            raster_names (list[str]): List of names corresponding to each raster file.
            logger: Logger object for logging progress and information.

        Returns:
            dict[tuple[int, int], dict[str, float]]: 
                Dictionary mapping (row, col) indices of suitable locations to a dictionary
                of constraint/cost values for each raster name.
        """
        t0 = time.time()
        logger.info('Collecting cost and constraint data for suitable siting locations...')

        suit_rows, suit_cols = np.where(suit_array == 1)
        suitrow_suitcol = list(zip(suit_rows, suit_cols))

        node_values: dict[tuple[int, int], dict[str, float]] = {}
        for node in suitrow_suitcol:
            node_values[node] = {}

        for path, name in zip(raster_paths, raster_names):
            logger.info(f"Loading {name} data from: {path}")
            raster_array = load_raster_array(path, window=window)
            logger.info(f"Assigning {name} values for {len(suitrow_suitcol)} suitable cells")
            for row, col in suitrow_suitcol:
                node_values[(row, col)][name] = raster_array[row, col]

        logger.info(
            f"All cost and constraint data loaded in {round(((time.time() - t0) / 60), 2)} minutes."
        )

        return node_values
