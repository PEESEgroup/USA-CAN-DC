import numpy as np
from scipy.ndimage import distance_transform_edt

def calc_gravity_array_from_distance(
        market_array: np.ndarray, 
        suit_array: np.ndarray, 
        beta:float=0.5
        ) -> np.ndarray:
    """
    Calculate the gravity multiplier array based on distance to the nearest market.
    Args:
        market_array (np.ndarray): 2D array representing market locations (1 for market,
            0 for non-market).
        suit_array (np.ndarray): 2D array representing siting suitability (1 for suitable,
            0 for unsuitable).
        beta (float): Exponent for market size.
    Returns:
        np.ndarray: 2D array of gravity multipliers.
    """

                                                       
    gravity_mask = np.where(market_array >0, 2, 0)

                                                  
    suitability_mask = np.where(suit_array == 1, -1, 0)

                       
    gravity_mask = gravity_mask + suitability_mask

                                                                                                  
    gravity_target = np.where(gravity_mask == -1, 0,
                            np.where(gravity_mask == 0, -1,
                                    1))

                                              
    target_mask = (gravity_target == 1)

                                                               
    distance_full, indices = distance_transform_edt(~target_mask, return_indices=True)

                                                                        
    output_distance = np.full_like(suit_array, np.nan, dtype=float)
    output_distance[suit_array == 1] = distance_full[suit_array == 1]

                                                                
    nearest_market_values = market_array[indices[0], indices[1]]

                            
    output_distance = (output_distance * 100) / 1000
    output_market_multiplier = output_distance / nearest_market_values

    return output_market_multiplier**beta


def calc_gravity_score(
        node: tuple, 
        gravity_multiplier_array: np.ndarray, 
        data_center_it_power_mw: float, 
        alpha:float=0.5
        ) -> float:

    """
    Calculate the gravity penalty based on distance, data center IT power, and market size.
    Args:
        node (tuple): Coordinates of the node (row, col).
        gravity_multiplier_array (np.ndarray): 2D array of gravity multipliers.
        data_center_it_power_mw (float): Data center IT power in megawatts.
        alpha (float): Exponent for data center IT power.
    Returns:
        float: Gravity score.
    """

                                                     
    market_gravity = gravity_multiplier_array[node[0], node[1]]

                                 
    gravity_score = (1 / (data_center_it_power_mw ** alpha)) * (market_gravity)

    return gravity_score