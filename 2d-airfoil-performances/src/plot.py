import json
import matplotlib.pyplot as plt

if __name__ == "__main__":

    airfoils_interp_data = json.load(open("../../data/airfoil_data_50000_interpolated.json"))
    airfoils_interp_names = list(airfoils_interp_data.keys())
    airfoils_orig_data = json.load(open("../../data/airfoil_data_50000.json"))
    airfoils_orig_names = list(airfoils_orig_data.keys())

    n = 1200

    plt.plot(
        *zip(*airfoils_orig_data[airfoils_orig_names[n]]["coords"]),
        "ro", label="Original", alpha=0.5
    )
    plt.plot(
        *zip(*airfoils_interp_data[airfoils_interp_names[n]]["coords"]),
        "bo", label="Interpolated", alpha=0.5
    )
    plt.legend()
    plt.show()