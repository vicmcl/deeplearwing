from pathlib import Path

import pandas as pd
import json
import numpy as np
import aerosandbox as asb



DATA_PATH = Path(__file__).parents[1] / "data"


def calculate_CST_parameters(name, x, y):
    coordinate_airfoil = asb.Airfoil(name, np.stack((x, y), axis=1))
    kulfan_airfoil = coordinate_airfoil.to_kulfan_airfoil()
    params = kulfan_airfoil.kulfan_parameters
    for key, value in params.items():
        if type(value) == np.ndarray:
            params[key] = value.tolist()
        if type(value) == np.float64:
            params[key] = float(value)
    return params


def json_extraction(file: Path):
    print(f"Processing {file.stem}")
    reynolds = file.stem.split('_')[-1]
    data = json.load(open(file, 'r'))
    return reynolds, data


def build_table(json_files: list[Path]):
    df = pd.DataFrame()

    for file in json_files:
        reynolds, data = json_extraction(file)
        rows = []

        for airfoil_name, airfoil_data in data.items():
            angles = airfoil_data['polars']['alpha']
            cd_values = airfoil_data['polars']['cd']
            cl_values = airfoil_data['polars']['cl']
            cm_values = airfoil_data['polars']['cm']
            x = airfoil_data['coords']['x']
            y = airfoil_data['coords']['y']

            params = calculate_CST_parameters(airfoil_name, x, y)

            angles_array = np.full(len(angles))
            combinations = np.column_stack((angles_array, reynolds), angles)

            for i, (reynolds, angle) in enumerate(combinations):
                rows.append({
                    'name': airfoil_name.replace('-il', ''),
                    'image': str(DATA_PATH / 'jpg' / f'{airfoil_name}.jpg'),
                    'angle': angle,
                    'reynolds': reynolds,
                    'x_coords': ' '.join(map(str, x)),
                    'y_coords': ' '.join(map(str, y)),
                    'cst_upper_weights': ' '.join(map(str, params['upper_weights'])),
                    'cst_lower_weights': ' '.join(map(str, params['lower_weights'])),
                    'cst_leading_edge_weight': params['leading_edge_weight'],
                    'cst_trailing_edge_thickness': params['TE_thickness'],
                    'cd': cd_values[i],
                    'cl': cl_values[i],
                    'cm': cm_values[i]
                })
        
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    return df


if __name__ == '__main__':

    json_files = [file for file in DATA_PATH.glob('json/*.json')]
    df = build_table(json_files)
    df = df[~df['name'].isin([
        'fx71l150',
        'rae69ck',
        'hq259b',
        'naca642215',
        'goe525',
        'nacam3',
        'goe801'
    ])]

    df.to_csv(DATA_PATH / 'csv' / 'airfoil_data.csv', index=False)
    df[df.columns.drop('image')].to_csv(
        DATA_PATH / 'csv' / 'DeepLearWing.csv', index=False
    )





