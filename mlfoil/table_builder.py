from pathlib import Path
import pandas as pd
import json
import numpy as np


DATA_PATH = Path(__file__).parents[1] / "data"

def build_table(json_files: list[Path]):
    df = pd.DataFrame()

    for file in json_files:
        print(f"Processing {file.stem}")
        reynolds = file.stem.split('_')[-1]
        data = json.load(open(file, 'r'))
        rows = []

        for airfoil_name, airfoil_data in data.items():
            angles = airfoil_data['polars']['alpha']
            cd_values = airfoil_data['polars']['cd']
            cl_values = airfoil_data['polars']['cl']
            cm_values = airfoil_data['polars']['cm']

            combinations = np.column_stack(
                (np.full(len(angles), reynolds), angles)
            )
            for i, (reynolds, angle) in enumerate(combinations):
                try:
                    rows.append({
                        'name': airfoil_name,
                        'image': str(DATA_PATH / 'jpg' / f'{airfoil_name}.jpg'),
                        'angle': angle,
                        'reynolds': reynolds,
                        'x_coords': ' '.join(map(str, airfoil_data['coords']['x'])),
                        'y_coords': ' '.join(map(str, airfoil_data['coords']['y'])),
                        'cd': cd_values[i],
                        'cl': cl_values[i],
                        'cm': cm_values[i]
                    })
                except IndexError:
                    print(f"Error: {airfoil_name} - {angle} - {reynolds}")
        
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    return df


if __name__ == '__main__':

    json_files = [file for file in DATA_PATH.glob('json/*.json')]
    df = build_table(json_files)
    df.to_csv(DATA_PATH / 'csv' / 'airfoil_data.csv', index=False)
    df[df.columns.drop('image')].to_csv(
        DATA_PATH / 'csv' / 'DeepLearWing.csv', index=False
    )





