import io
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

if 'google.colab' in sys.modules:
    from src.features import string_to_floats
else:
    from features import string_to_floats


DATA_PATH = Path(__file__).parents[2] / "data"


def coords_to_image(x, y):
    fig = create_figure(x, y)
    y_range = calculate_y_range(y)
    update_layout(fig, y_range)
    img = generate_image(fig)
    return img


def create_figure(x, y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, 
        y=y, 
        fill='toself', 
        fillcolor='black', 
        line=dict(color='black')
    ))
    return fig


def calculate_y_range(y):
    max_y = max(abs(np.max(y)), abs(np.min(y)))
    padding = 0.1 * max_y
    y_range = [-max_y - padding, max_y + padding]
    return y_range


def update_layout(fig, y_range):
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            range=y_range,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        autosize=False,
        width=224,
        height=224,
    )
    fig.update_yaxes(constrain='domain')


def generate_image(fig):
    img_bytes = fig.to_image(format="jpg")
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert('L')  # Convert image to grayscale
    img_array = np.array(img)
    return img_array


if __name__=='__main__':

    df = pd.read_csv(DATA_PATH / 'csv' / 'DeepLearWing.csv')
    x = string_to_floats(df.iloc[0, :]['x_coords'])
    y = string_to_floats(df.iloc[0, :]['y_coords'])
    plt.imshow(coords_to_image(x, y))
    plt.show()