import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import IMG_SIZE, LR


def se_block(x, reduction=16):
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(channels // reduction, activation="relu")(se)
    se = layers.Dense(channels, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, channels))(se)
    return layers.Multiply()([x, se])


def build_model(img_size=IMG_SIZE, lr=LR):
    inputs  = keras.Input(shape=(img_size[0], img_size[1], 3))
    base    = tf.keras.applications.EfficientNetB0(
        include_top  = False,
        weights      = "imagenet",
        input_tensor = inputs
    )
    base.trainable = False
    x       = base.output
    x       = se_block(x)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model   = keras.Model(inputs, outputs)
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=lr),
        loss      = "binary_crossentropy",
        metrics   = ["accuracy", keras.metrics.AUC(name="auc")]
    )
    return model