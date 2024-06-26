import aerosandbox as asb
import aerosandbox.numpy as np
import json

for reynolds in [50_000, 100_000, 200_000, 500_000, 1_000_000]:
    airfoils_data = json.load(
        open(f"../data/airfoil_data_{reynolds}.json", "r")
    )
    airfoil_names = list(airfoils_data.keys())

    for name in airfoil_names:
        airfoil = airfoils_data[name]
        coords = airfoil['coords']
        x = np.array(coords['x'])
        y = np.array(coords['y'])
        coordinate_airfoil = asb.Airfoil(name, np.stack((x, y), axis=1))
        kulfan_airfoil = coordinate_airfoil.to_kulfan_airfoil()
        params = kulfan_airfoil.kulfan_parameters

        for key, value in params.items():
            if type(value) == np.ndarray:
                params[key] = value.tolist()
            if type(value) == np.float64:
                params[key] = float(value)

        airfoil['cts'] = params
    with open(f"../data/airfoil_data_{reynolds}.json", "w") as f:
        json.dump(airfoils_data, f, indent = 4)