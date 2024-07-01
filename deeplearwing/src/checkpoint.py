from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path

DATA_PATH = Path(__file__).parents[2] / "data"

def checkpoint(fname, patience = 5):
    weights_path = DATA_PATH / 'weights'
    callbacks = [
        EarlyStopping(patience = patience),    # Stop the training after 20 iterations without improvement
        ModelCheckpoint(                        # Save the weights giving the best results in a file
            filepath = str(weights_path / fname),
            monitor = "val_loss",
            mode = 'min',
            save_best_only = True,
        )
    ]

    return callbacks