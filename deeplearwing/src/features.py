import aerosandbox as asb
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parents[1] / "data"


def string_to_floats(item: list | str, reverse = False):
    # List of floats to string
    if reverse:
        return ' '.join([str(x) for x in item])
    
    # String to list of floats
    return [float(x) for x in item.split()]


# Define a function to calculate CST parameters for an airfoil
def calculate_CST_params(name, x, y):
    """
    Calculate the CST (Class Shape Transformation) parameters for a given airfoil.

    Parameters:
    - name (str): The name of the airfoil.
    - x (array-like): The x-coordinates of the airfoil points.
    - y (array-like): The y-coordinates of the airfoil points.

    Returns:
    - dict: A dictionary containing the CST parameters of the airfoil.

    """
    # Create an Airfoil object from the given coordinates
    coordinate_airfoil = asb.Airfoil(name, np.stack((x, y), axis=1))

    # Convert the coordinate-based airfoil to a Kulfan airfoil
    kulfan_airfoil = coordinate_airfoil.to_kulfan_airfoil()

    # Get the CST parameters from the Kulfan airfoil
    params = kulfan_airfoil.kulfan_parameters

    # Convert any numpy arrays to strings and any np.float64 values to floats
    for key, value in params.items():
        if type(value) == np.ndarray:
            params[key] = string_to_floats(value, reverse=True)
        if type(value) == np.float64:
            params[key] = float(value)
            
    # Return the CST parameters as a dictionary
    return {name: params}


def assign_cst_params(df, cst_params):
    """
    Assigns CST (Class Shape Transformation) parameters to a DataFrame based on the 'name' column.

    Args:
        df (pandas.DataFrame): The DataFrame to which the CST parameters will be assigned.
        cst_params (dict): A dictionary containing CST parameters for different 'name' values.

    Returns:
        pandas.DataFrame: The DataFrame with CST parameters assigned.

    Raises:
        KeyError: If 'name' value is not found in the cst_params dictionary.

    Example:
        cst_params = {
            'name1': {
                'upper_weights': [0.1, 0.2, 0.3],
                'lower_weights': [0.3, 0.2, 0.1],
                'leading_edge_weight': 0.5,
                'TE_thickness': 0.01
            },
            'name2': {
                'upper_weights': [0.2, 0.3, 0.4],
                'lower_weights': [0.4, 0.3, 0.2],
                'leading_edge_weight': 0.6,
                'TE_thickness': 0.02
            },
            ...
        }

        df = assign_cst_params(df, cst_params)
    """
    # Map CST parameters to DataFrame based on 'name' column
    df = df.assign(
        upper_weights = df['name'].map(
            lambda x: cst_params.get(x).get('upper_weights')
        ),
        lower_weights = df['name'].map(
            lambda x: cst_params.get(x).get('lower_weights')
        ),
        leading_edge_weight = df['name'].map(
            lambda x: cst_params.get(x).get('leading_edge_weight')
        ),
        trailing_edge_thickness = df['name'].map(
            lambda x: cst_params.get(x).get('TE_thickness')
        ),
    )
    return df


if __name__ == '__main__':

    df = pd.read_csv(DATA_PATH / 'csv' / 'data.csv')
    df = remove_duplicate_airfoils(df)

    # Create an empty dictionary to store CST parameters
    dict_cst = {}

    # Iterate over unique names in the DataFrame
    for name in df['name'].unique():
        # Filter DataFrame for the current name
        df_name = df[df['name'] == name]
        
        # Convert x_coords and y_coords to lists of floats
        x_coords = string_to_floats(df_name['x_coords'].iloc[0])
        y_coords = string_to_floats(df_name['y_coords'].iloc[0])
        
        # Calculate CST parameters for the current airfoil
        cst_params = calculate_CST_params(name, x_coords, y_coords)
        
        # Update the dictionary with the CST parameters
        dict_cst.update(cst_params)

    df = assign_cst_params(df, dict_cst)
    print(df.head())