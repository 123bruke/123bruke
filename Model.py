from tensorflow import keras
from tensorflow.keras import layers

def build_model(normalizer):
    model = keras.Sequential([
        normalizer,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error']
    )

    return model
