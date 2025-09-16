import os
import pickle
from sklearn.pipeline import Pipeline
from transformer import ColumnDropper, EncodeAndScale, KerasWrapper
from tensorflow import keras

PICKLE_DIR = os.path.join(os.path.dirname(__file__), "pickle_files")

with open(os.path.join(PICKLE_DIR, "encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

with open(os.path.join(PICKLE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# load keras model properly
model = keras.models.load_model("pickle_files/model.keras")

deployment_pipeline = Pipeline([
    ("dropper", ColumnDropper(drop_cols=["RowNumber", "CustomerId", "Surname"])),
    ("encoder_scaler", EncodeAndScale(encoder, scaler)),
    ("model", KerasWrapper(model))
])

with open(os.path.join(PICKLE_DIR, "final_pipeline.pkl"), "wb") as f:
    pickle.dump(deployment_pipeline, f)

print("✅ Final pipeline saved.")
