from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    concatenate,
    MaxPooling2D,
    BatchNormalization,
    Activation,
    Dropout,
)
from keras.optimizers import Adam
from utils import psnr, ssim


def conv_block(x, filters, kernel_size=(3, 3), activation="relu", padding="same"):
    x = Conv2D(filters, kernel_size, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(filters, kernel_size, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x


def upsample_block(
    x, skip_connection, filters, kernel_size=(3, 3), strides=(2, 2), padding="same"
):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = concatenate([skip_connection, x])
    x = Dropout(0.5)(x)
    x = conv_block(x, filters)
    return x


def build_encoder(x, filters, levels=5):
    skips = []

    for _ in range(levels):
        x = conv_block(x, filters)
        skips.append(x)
        x = MaxPooling2D((2, 2))(x)

    return x, skips


def build_decoder(x, skips, filters, levels=5):
    for i in reversed(range(levels)):
        x = upsample_block(x, skips[i], filters)

    return x


def autoencoder(optimizer=Adam(lr=1e-4), input_shape=(128, 128, 3)):
    filters = 64
    input_img = Input(input_shape, name="image_input")

    x, skips = build_encoder(input_img, filters)
    x = conv_block(x, filters)
    x = build_decoder(x, skips, filters)

    # Output Layer
    output_layer = Conv2D(3, (3, 3), padding="same", activation="sigmoid")(x)

    autoencoder = Model(inputs=input_img, outputs=output_layer)
    autoencoder.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=[ssim, psnr],
    )

    return autoencoder