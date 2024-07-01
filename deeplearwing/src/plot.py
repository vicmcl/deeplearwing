import json
import numpy as np
import plotly.graph_objects as go

from pathlib import Path

DATA_PATH = Path(__file__).parents[1] / "data"


def main():

    airfoils_data = json.load(
        open(DATA_PATH / 'json' / 'airfoil_data_1000000.json', 'r')
    )

    for airfoil in airfoils_data.keys():
        x = airfoils_data[airfoil]['coords']['x']
        y = airfoils_data[airfoil]['coords']['y']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, 
            y=y, 
            fill='toself', 
            fillcolor='black', 
            line=dict(color='black')
        ))

        # Calculate the maximum absolute y-coordinate
        max_y = max(abs(np.max(y)), abs(np.min(y)))
        
        # Add a small padding (e.g., 10% of max_y)
        padding = 0.1 * max_y
        y_range = [-max_y - padding, max_y + padding]

        # Update layout
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(
                showgrid = False,
                zeroline = False,
                showline = False,
                showticklabels = False,
                scaleanchor = "y",
                scaleratio = 1,
            ),
            yaxis = dict(
                showgrid = False,
                zeroline = False,
                showline = False,
                showticklabels = False,
                range = y_range,  # Set fixed y-axis range
            ),
            margin = dict(l = 0, r = 0, t = 0, b = 0),
            autosize = False,
            width = 800,  # Adjust as needed
            height = 400,  # Adjust as needed
        )

        # Ensure the aspect ratio is maintained
        fig.update_yaxes(constrain='domain')
        fig.show()

        
if __name__ == "__main__":
    main()