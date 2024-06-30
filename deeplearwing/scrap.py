import requests
import pandas as pd
import json

from pathlib import Path
from bs4 import BeautifulSoup


def get_polars(airfoil_name, reynolds):
    """
    Retrieves polar data for a given airfoil and Reynolds number from airfoiltools.com.

    Args:
        airfoil_name (str): The name of the airfoil.
        reynolds (int): The Reynolds number.

    Returns:
        list: A list of lists representing the polar data, where each inner list contains the values for a specific angle of attack.

    Raises:
        Exception: If the request to airfoiltools.com fails or if the HTML parsing fails.
    """
    url = f"http://airfoiltools.com/polar/details?polar=xf-{airfoil_name}-{reynolds}"
    response = requests.get(url)
    soup = BeautifulSoup(
        response.text, "html.parser").find("td", class_="cell2"
    ).text
    lines = [l.strip() for l in soup.splitlines() if l.strip() != ""][5:]
    columns = lines[0].split()
    data = pd.DataFrame([l.split() for l in lines[2:]], columns=columns).apply(pd.to_numeric)
    return data.to_numpy().tolist()


def get_coords(airfoil_name):
    """
    Retrieves the coordinates of an airfoil from the Airfoil Tools website.

    Parameters:
    - airfoil_name (str): The name of the airfoil.

    Returns:
    - list: A list of coordinate pairs representing the airfoil's shape.

    Example:
    >>> get_coords("NACA0012")
    [[1.000, 0.000], [0.998, 0.024], [0.993, 0.048], ...]
    """
    url = f"http://airfoiltools.com/airfoil/seligdatfile?airfoil={airfoil_name}"
    response = requests.get(url)
    data = BeautifulSoup(response.text, "html.parser").text.splitlines()[1:]
    df = pd.DataFrame([l.split() for l in data], columns=["x", "y"]).apply(pd.to_numeric)
    return df.to_numpy().tolist()


def get_all_airfoils():
    """
    Retrieves a list of all airfoils from the airfoiltools.com website.

    Returns:
        airfoils (list): A list of airfoil names.

    """
    url = "http://airfoiltools.com/search/airfoils"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser") \
           .find("table", class_="listtable") \
           .find_all("a")
    airfoils = [af.get('href').split('=')[1] for af in soup]
    return airfoils


def get_data(reynolds):
    airfoil_names = get_all_airfoils()
    data = {}
    for airfoil_name in airfoil_names:
        try:
            coords = get_coords(airfoil_name)
            polars = get_polars(airfoil_name, reynolds)
            x, y = map(list, zip(*coords))
            dict_coords = {"x": x, "y": y}
            alpha, cl, cd, cdp, cm, otp_xtr, bot_xtr = map(list, zip(*polars))
            dict_polars = {
                "alpha": alpha,
                "cl": cl,
                "cd": cd,
                "cdp": cdp,
                "cm": cm,
                "otp_xtr": otp_xtr,
                "bot_xtr": bot_xtr
            }
            data[airfoil_name] = {"coords": dict_coords, "polars": dict_polars}
        except Exception as e:
            print(f"Failed to fetch data for {airfoil_name}: {e}")
    return data


if __name__ == "__main__":

    DATA_PATH = Path(__file__).parents[1] / "data"
    
    # Iterate over different Reynolds numbers
    for reynolds in [50_000, 100_000, 200_000, 500_000, 1_000_000]:

        # Get airfoil data for the current Reynolds number
        json_data = get_data(reynolds)

        # Save the data as JSON file
        with open(DATA_PATH / f"airfoil_data_{reynolds}.json", "w") as f:
            json.dump(json_data, f, indent=4)