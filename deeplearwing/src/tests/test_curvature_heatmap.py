from deeplearwing.src.curvature_heatmap import compute_curvature
from deeplearwing.src.features import string_to_floats
from deeplearwing.src.plot import airfoil_to_image
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path(__file__).parents[3] / "data"


if __name__ == "__main__":
    df_csv = pd.read_csv(DATA_PATH / "csv" / "DeepLearWing.csv")
    row = df_csv.sample(1, random_state=42)
    x = string_to_floats(row["x_coords"].iloc[0])
    y = string_to_floats(row["y_coords"].iloc[0])
    img = airfoil_to_image(x, y, 512, 256)
    heatmap = compute_curvature(img, smooth_value=3)

    fig = plt.figure(figsize=(12, 6))
    fig.add_subplot(2, 1, 1)
    plt.imshow(heatmap, cmap="binary")
    fig.add_subplot(2, 1, 2)
    plt.imshow(img, cmap="binary_r")
    plt.tight_layout()
    plt.show()
