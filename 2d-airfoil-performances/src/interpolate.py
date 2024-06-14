import json
import numpy as np
import matplotlib.pyplot as plt


def split_suction_pressure_sides(x, y):
    idx_split = list(x).index(min(x))
    x_suction, x_pressure = x[:idx_split], x[idx_split:]
    y_suction, y_pressure = y[:idx_split], y[idx_split:]
    x_suction, y_suction = x_suction[::-1], y_suction[::-1]
    return x_suction, y_suction, x_pressure, y_pressure


def get_leading_edge_points(x, y):
    leading_edge_points = []
    for i in range(1, len(x)):
        if x[i] - x[i-1] < 0.01:
            leading_edge_points.append([x[i-1], y[i-1]])
        else:
            break
    return leading_edge_points


def get_max_points(data, airfoil_names):
    max_points = 0
    for airfoil_name in airfoil_names:
        l = len(data[airfoil_name]["coords"])
        if l > max_points:
            max_points = l
    return max_points


def interpolate_coords(x_orig, y_orig, num_points):
    new_coords = []
    leading_edge_points = get_leading_edge_points(x_orig, y_orig)
    idx_min = len(leading_edge_points)
    try:
        new_x = np.linspace(x_orig[idx_min], x_orig[-1], num_points - idx_min)
        new_y = np.interp(new_x, x_orig[idx_min:], y_orig[idx_min:])
        new_coords += zip(new_x, new_y)
        new_coords += leading_edge_points
        new_coords = sorted(new_coords, key=lambda x: x[0])
        return new_coords
    except:
        return None


def interpolated_airfoil(airfoil_name, num_points):
    coord_list = data[airfoil_name]["coords"]
    x, y = np.array(coord_list).T
    x_suction, y_suction, x_pressure, y_pressure = split_suction_pressure_sides(x, y)
    new_coords = []
    for x_orig, y_orig in [(x_suction, y_suction), (x_pressure, y_pressure)]:
        if len(coord_list) < num_points:
            interp_coords = interpolate_coords(x_orig, y_orig, num_points // 2)
            if interp_coords is None:
                new_coords = None
                break
            new_coords += interp_coords
        else:
            new_coords += list(zip(x_orig, y_orig))
    return new_coords

    
if __name__ == "__main__":

    data = json.load(open("../../data/airfoil_data_50000.json"))
    airfoil_names = list(data.keys())
    num_points = get_max_points(data, airfoil_names)

    for airfoil_name in airfoil_names:
        new_coords = interpolated_airfoil(airfoil_name, num_points)
        if new_coords is not None:
            data[airfoil_name]["coords"] = new_coords
        else:
            print(f"Error interpolating {airfoil_name}")

    json.dump(
        data,
        open("../../data/airfoil_data_50000_interpolated.json", "w"),
        indent=4
    )

        
       
        
