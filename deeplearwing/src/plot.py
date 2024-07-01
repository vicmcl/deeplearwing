import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

from pathlib import Path
from PIL import Image, ImageDraw

if 'google.colab' in sys.modules:
    from src.features import string_to_floats
else:
    from features import string_to_floats


DATA_PATH = Path(__file__).parents[2] / "data"

import numpy as np
from PIL import Image, ImageDraw
from typing import List

def airfoil_to_image(x: List[float], y: List[float], image_width: int = 512, image_height: int = 128) -> np.ndarray:
    """
    Convert airfoil coordinates to a black and white image representation.
    
    Args:
    x (List[float]): List of x-coordinates of the airfoil.
    y (List[float]): List of y-coordinates of the airfoil.
    image_width (int): Width of the output image.
    image_height (int): Height of the output image.
    
    Returns:
    np.ndarray: NumPy array of shape (image_height, image_width) representing the airfoil image.
    """
    assert len(x) == len(y), "x and y must have the same length"
    normalized_coords = normalize_coordinates(x, y, image_width, image_height)
    img = create_white_image(image_width, image_height)
    draw_airfoil(img, normalized_coords)
    return np.array(img)


def normalize_coordinates(x: List[float], y: List[float], image_width: int, image_height: int) -> List[Tuple[int, int]]:
    """
    Normalize the airfoil coordinates to fit within the image dimensions.
    
    Args:
    x (List[float]): List of x-coordinates of the airfoil.
    y (List[float]): List of y-coordinates of the airfoil.
    image_width (int): Width of the output image.
    image_height (int): Height of the output image.
    
    Returns:
    List[Tuple[int, int]]: List of normalized coordinates as tuples of (x, y).
    """
    # Ensure x and y have the same length
    assert len(x) == len(y), "x and y must have the same length"
    
    # Calculate the minimum and maximum values of x and y
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    
    # Calculate scaling factors for x and y separately
    x_scale = (image_width * 1) / (x_max - x_min)
    y_scale = (image_height * 1) / (y_max - y_min)
    
    # Use the smaller scale to ensure the entire airfoil fits
    scale = min(x_scale, y_scale)
    
    # Calculate the x and y offsets
    x_offset = (image_width - scale * (x_max - x_min)) / 2
    y_offset = (image_height - scale * (y_max - y_min)) / 2
    
    # Normalize the coordinates
    normalized_coords = [
        (int(x_offset + scale * (xi - x_min)), 
         int(y_offset + scale * (yi - y_min)))
        for xi, yi in zip(x, y)
    ]
    
    return normalized_coords


def create_white_image(image_width: int, image_height: int) -> Image:
    """
    Create a white image with the specified dimensions.
    
    Args:
    image_width (int): Width of the image.
    image_height (int): Height of the image.
    
    Returns:
    Image: White image.
    """
    return Image.new('L', (image_width, image_height), color=255)


def draw_airfoil(img: Image, normalized_coords: List[Tuple[int, int]]) -> None:
    """
    Draw the airfoil on the image.
    
    Args:
    img (Image): Image to draw on.
    normalized_coords (List[Tuple[int, int]]): List of normalized coordinates as tuples of (x, y).
    """
    draw = ImageDraw.Draw(img)
    draw.polygon(normalized_coords, fill=0)


if __name__=='__main__':

    df = pd.read_csv(DATA_PATH / 'csv' / 'DeepLearWing.csv')
    x = string_to_floats(df.iloc[1000, :]['x_coords'])
    y = string_to_floats(df.iloc[1000, :]['y_coords'])
    plt.imshow(airfoil_to_image(x, y))
    plt.show()