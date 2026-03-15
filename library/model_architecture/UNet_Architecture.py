from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, MaxPooling2D
from utils import psnr, ssim


def conv_block(x, filters, kernel_size=(3, 3), activation="relu", padding="same"):
    x = Conv2D(filters, kernel_size, activation=activation, padding=padding)(x)
    x = Conv2D(filters, kernel_size, activation=activation, padding=padding)(x)
    return x


def upsample_block(
    x,
    skip_connection,
    filters,
    kernel_size=(3, 3),
    activation="relu",
    strides=(2, 2),
    padding="same",
):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = concatenate([skip_connection, x])
    x = conv_block(x, filters, activation=activation)
    return x


def build_encoder(x, filters, levels, activation="relu"):
    skips = []
    for _ in range(levels):
        x = conv_block(x, filters, activation=activation)
        skips.append(x)
        x = MaxPooling2D((2, 2))(x)
    return x, skips


def build_decoder(x, skips, filters, levels, activation="relu"):
    for i in reversed(range(levels)):
        x = upsample_block(x, skips[i], filters, activation=activation)
    return x


def unet(
    optimizer, input_shape=(None, None, 3), filters=128, levels=5, activation="relu"
):
    input_img = Input(shape=input_shape, name="image_input")
    x, skips = build_encoder(input_img, filters, levels, activation=activation)
    x = conv_block(x, filters, activation=activation)
    x = build_decoder(x, skips, filters, levels, activation=activation)
    output_layer = Conv2D(3, (3, 3), padding="same", activation="sigmoid")(x)

    model = Model(inputs=input_img, outputs=output_layer)
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=[ssim, psnr])
    return model
