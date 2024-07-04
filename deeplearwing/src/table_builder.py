from pathlib import Path

import pandas as pd
import json
import numpy as np


DATA_PATH = Path(__file__).parents[1] / "data"


def json_ingestion(file: Path):
    """
    Ingests a JSON file and returns the Reynolds number and the data.

    Parameters:
    file (Path): The path to the JSON file.

    Returns:
    tuple: A tuple containing the Reynolds number (extracted from the file name) and the loaded data from the JSON file.
    """
    print(f"Processing {file.stem}")
    reynolds = file.stem.split("_")[-1]
    data = json.load(open(file, "r"))
    return reynolds, data


def safe_get(data: dict, dot_chained_keys: str):
    """
    Safely retrieves a value from a nested dictionary using dot-chained keys.

    Args:
        data (dict): The nested dictionary to retrieve the value from.
        dot_chained_keys (str): The dot-chained keys representing the path to the desired value.

    Returns:
        The value retrieved from the nested dictionary, or None if any of the keys are not found.

    """
    keys = dot_chained_keys.split(".")
    for key in keys:
        try:
            if isinstance(data, list):
                data = data[0][key]
            else:
                data = data[key]
        except (KeyError, TypeError, IndexError):
            return None
    return data


def build_table(json_files: list[Path]):
    """
    Builds a table from a list of JSON files.

    Args:
        json_files (list[Path]): A list of JSON file paths.

    Returns:
        pd.DataFrame: The resulting table containing data from the JSON files.
    """
    df = pd.DataFrame()
    for file in json_files:
        reynolds, airfoil_data = json_ingestion(file)
        rows = []

        for airfoil_name, data in airfoil_data.items():
            angles = safe_get(data, "polars.alpha")
            reynolds_array = np.full(len(angles), reynolds)
            combinations = np.column_stack((reynolds_array, angles))

            for i, (reynolds, angle) in enumerate(combinations):
                rows.append(
                    {
                        "name": airfoil_name.replace("-il", ""),
                        "angle": angle,
                        "reynolds": reynolds,
                        "x_coords": " ".join(map(str, safe_get(data, "coords.x"))),
                        "y_coords": " ".join(map(str, safe_get(data, "coords.y"))),
                        "cd": safe_get(data, "polars.cd")[i],
                        "cl": safe_get(data, "polars.cl")[i],
                        "cm": safe_get(data, "polars.cm")[i],
                    }
                )

        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return df


if __name__ == "__main__":

    json_files = [file for file in DATA_PATH.glob("json/*.json")]
    df = build_table(json_files)
    df.to_csv(DATA_PATH / "csv" / "data.csv", index=False)
