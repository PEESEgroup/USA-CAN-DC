from collections import deque
from math import ceil, isqrt
from typing import List, Dict, Any
from affine import Affine

import rasterio
import numpy as np
import networkx as nx


def get_region_suit_array(
    region_array: np.ndarray,
    suit_array: np.ndarray,
    region_id: int
) -> np.ndarray:
    """
    Return the suitability array for a specific region.

    Args:
        region_array (np.ndarray): 
            A 2D numpy array representing region IDs for each grid cell.
        suit_array (np.ndarray): 
            A 2D numpy array representing the suitability value of each grid cell.
        region_id (int): 
            The ID of the region for which to extract the suitability array.

    Returns:
        np.ndarray: 
            A 2D numpy array where only the cells belonging to the specified region
            retain their suitability values, and all other cells are set to zero.

    Raises:
        ValueError: If the specified region_id is not present in region_array.
    """
    if region_id not in np.unique(region_array):
        raise ValueError(f"Region ID {region_id} not found in the region raster.")

                                            
    region_mask = np.where(region_array == region_id, 1, 0)

                                             
    region_suit_array = suit_array * region_mask

    return region_suit_array


def get_connected_nodes(
    graph: nx.Graph,
    start_node,
    min_block_size: int
) -> list:
    """
    Perform a breadth-first search (BFS) to find a set of connected nodes in a graph,
    starting from a specified node, and return at least `min_block_size` nodes.

    Args:
        graph (nx.Graph): The graph in which to search for connected nodes.
        start_node: The node from which to start the search.
        min_block_size (int): The minimum number of connected nodes to return.

    Returns:
        list: A list of nodes representing a connected component containing at least
            `min_block_size` nodes, starting from `start_node`.

    Raises:
        ValueError: If `start_node` is not present in the graph, or if there are not
            enough connected nodes reachable from `start_node` to satisfy `min_block_size`.
    """
    if start_node not in graph:
        raise ValueError("Start node not in graph.")

    visited = set()
    queue = deque([start_node])
    result = []

                                                          
    while queue and len(result) < min_block_size:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            result.append(node)
            queue.extend(n for n in graph.neighbors(node) if n not in visited)

    if len(result) < min_block_size:
        raise ValueError("Not enough connected nodes from the starting node.")

    return result


def _factor_pairs(cell_count: int) -> list[tuple[int, int]]:
    """Return rectangle dimensions for ``cell_count``, least elongated first."""
    pairs = []
    for height in range(1, isqrt(cell_count) + 1):
        if cell_count % height == 0:
            width = cell_count // height
            pairs.append((height, width))
            if height != width:
                pairs.append((width, height))
    return sorted(pairs, key=lambda dimensions: (abs(dimensions[0] - dimensions[1]), dimensions))


def find_nearest_rectangle(
    graph: nx.Graph,
    start_node: tuple[int, int],
    target_cell_count: int,
    max_aspect_ratio: float = 5.0,
    max_area_growth_ratio: float = 0.5,
) -> list[tuple[int, int]] | None:
    """Find the smallest adequate all-suitable rectangle containing ``start_node``.

    Candidate areas are checked from ``target_cell_count`` upwards, so an
    undersized parcel is never selected.  To keep site selection tractable,
    the search stops at a 50% area increase (or five cells, whichever is
    larger).  If no rectangle in that practical range contains the current
    lowest-scoring cell, the caller moves on to the next cell.  Only rectangles
    with a length-to-width ratio at or below ``max_aspect_ratio`` are
    considered.  For an equal area, compact rectangles are preferred.
    """
    if target_cell_count < 1:
        raise ValueError("target_cell_count must be at least 1.")
    if int(target_cell_count) != target_cell_count:
        raise ValueError("target_cell_count must be a whole number of grid cells.")
    if max_aspect_ratio < 1:
        raise ValueError("max_aspect_ratio must be at least 1.")
    if max_area_growth_ratio < 0:
        raise ValueError("max_area_growth_ratio must not be negative.")
    target_cell_count = int(target_cell_count)
    if start_node not in graph:
        return None

    available_nodes = set(graph.nodes)
    max_cell_count = min(
        len(available_nodes),
        target_cell_count + max(5, ceil(target_cell_count * max_area_growth_ratio)),
    )
    row, col = start_node

    for cell_count in range(target_cell_count, max_cell_count + 1):
        for height, width in _factor_pairs(cell_count):
            if max(height, width) / min(height, width) > max_aspect_ratio:
                continue
                                                                      
                                                                
            for top_row in range(row - height + 1, row + 1):
                for left_col in range(col - width + 1, col + 1):
                    rectangle = [
                        (rectangle_row, rectangle_col)
                        for rectangle_row in range(top_row, top_row + height)
                        for rectangle_col in range(left_col, left_col + width)
                    ]
                    if all(node in available_nodes for node in rectangle):
                        return rectangle
    return None


def build_graph(
    region_suit_array: "np.ndarray",
    min_block_size: int,
    raster_names: list[str],
    node_values: dict[tuple[int, int], dict[str, float]]
) -> "nx.Graph":
    """
    Build a graph from the region suitability array, where each node represents a suitable grid cell
    and edges connect adjacent suitable cells. Node attributes are populated with raster values.

    Args:
        region_suit_array (np.ndarray): 2D numpy array representing the suitability of each grid cell in the region.
                                        Suitable cells should have a value of 1.
        min_block_size (int): Minimum number of connected nodes (grid cells) required for a site to be considered valid.
        raster_names (list[str]): List of raster data names to be used as node attributes.
        node_values (dict[tuple[int, int], dict[str, float]]): Dictionary mapping (row, col) tuples to dictionaries
                                                               of raster names and their corresponding values.

    Returns:
        nx.Graph: A NetworkX graph where nodes represent suitable grid cells and edges represent connectivity
                  between adjacent suitable cells. Node attributes are populated with raster values.
    """
    rows, cols = np.where(region_suit_array == 1)
    one_pixels = set(zip(rows, cols))

    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    G = nx.Graph()

                         
    for row, col in one_pixels:
        G.add_node((row, col))
        for dr, dc in neighbor_offsets:
            neighbor = (row + dr, col + dc)
            if neighbor in one_pixels:
                G.add_edge((row, col), neighbor)

    for component in list(nx.connected_components(G)):
        if len(component) < min_block_size:
            G.remove_nodes_from(component)

                                       
    for name in raster_names:
        for node in G.nodes:
            G.nodes[node][name] = node_values[node][name]

    return G


def site_based_on_siting_score(
    G: nx.Graph,
    number_of_sites: int | None,
    min_block_size: int,
    region_name: str,
    transform: Affine,
    attribute: str | None = 'total_weighted_siting_score'
) -> List[Dict[int, Dict[str, Any]]]:
    """
    Select sites based on the minimum locational cost and gravity score from a graph of suitable areas.

    Args:
        G (nx.Graph): The graph representing the siting areas, where nodes are grid cells and
            node attributes include locational cost and other relevant data.
        number_of_sites (int): The desired number of sites to be selected.
        min_block_size (int): The minimum number of connected nodes required to consider a site valid.
        region_name (str): The name of the region for which sites are being selected.
        transform (Affine): The affine transformation to convert pixel coordinates to geographic coordinates.
        attribute (str, optional): The attribute in the graph nodes that contains the locational cost.
            Defaults to 'locational_cost'.

    Returns:
        List[Dict[int, Dict[str, Any]]]: A list of dictionaries, each containing information about a selected site.
            Each dictionary has a single key (site index) mapping to a dictionary with the following keys:
                - 'region_name' (str): Name of the region.
                - 'min_node' (Tuple[float, float]): Geographic coordinates (x, y) of the minimum cost node.
                - 'locational_cost' (float): Locational cost value for the site.
                - 'coord_list' (List[Tuple[float, float]]): List of geographic coordinates (x, y) for the selected nodes.
                - 'row_col_list' (List[Tuple[int, int]]): List of (row, col) indices for the selected nodes.
    """
                                                                
    H = G.copy()
    i = 0

    result_list: List[Dict[int, Dict[str, Any]]] = []

                                                                                                
    while (number_of_sites is None or len(result_list) < number_of_sites) and H.number_of_nodes() > 0:
        try:
            if attribute is None:
                min_node = min(H.nodes)
            else:
                min_node = min(
                    (node for node, data in H.nodes(data=True) if attribute in data),
                    key=lambda node: H.nodes[node][attribute]
                )
        except ValueError:
                           
            break

                                                                             
                                                                              
                                                                           
        connections = nx.node_connected_component(H, min_node)
        selected_rectangle = (
            find_nearest_rectangle(H, min_node, min_block_size)
            if len(connections) >= min_block_size
            else None
        )

        if selected_rectangle is not None:
                                                                             
                                                                              
            nodes_to_remove = selected_rectangle

            result_dict: Dict[int, Dict[str, Any]] = {}

                           
            row, col = min_node
            x, y = rasterio.transform.xy(transform, row, col)

                               
            coord_list = []
            row_col_list = []
            for neighbor in selected_rectangle:
                row_n, col_n = neighbor
                x_n, y_n = rasterio.transform.xy(transform, row_n, col_n)
                coord_list.append((x_n, y_n))
                row_col_list.append((row_n, col_n))

            result_dict[i] = {
                'region_name': region_name,
                'min_node': (x, y),
                'weighted_siting_score': H.nodes[min_node].get(attribute) if attribute else None,
                'locational_cost': H.nodes[min_node].get('locational_cost'),
                'normalized_locational_cost': H.nodes[min_node].get('normalized_locational_cost'),
                'normalized_gravity_score': H.nodes[min_node].get('normalized_gravity_score'),
                'coord_list': coord_list,
                'row_col_list': row_col_list,
                'selected_grid_cell_count': len(selected_rectangle),
            }

                                                  
            result_dict[i].update(H.nodes[min_node]['parameters'])
            
                                               
            result_list.append(result_dict)

                                               
            H.remove_nodes_from(nodes_to_remove)
            i += 1

        else:
                                                                                    
            H.remove_node(min_node)

    return result_list
