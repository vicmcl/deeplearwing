import requests
import pandas as pd
import json
from bs4 import BeautifulSoup


def get_polars(airfoil_name, reynolds):
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
    url = f"http://airfoiltools.com/airfoil/seligdatfile?airfoil={airfoil_name}"
    response = requests.get(url)
    data = BeautifulSoup(response.text, "html.parser").text.splitlines()[1:]
    df = pd.DataFrame([l.split() for l in data], columns=["x", "y"]).apply(pd.to_numeric)
    return df.to_numpy().tolist()


def get_all_airfoils():
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
            data[airfoil_name] = {"coords": coords, "polars": polars}
        except Exception as e:
            print(f"Failed to fetch data for {airfoil_name}: {e}")
    return data


if __name__ == "__main__":
    json_data = get_data(50_000)
    with open("airfoil_data_50000.json", "w") as f:
        json.dump(json_data, f, indent=4)