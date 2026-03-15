import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageFont
import visualkeras
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping


def preprocess_and_load_data(directory, file_names, target_size=(128, 128)):
    """
    Preprocesses and loads images from a directory.

    Args:
        directory (str): Path to the directory containing images.
        file_names (list): List of image filenames.
        target_size (tuple): Target size for resizing images.

    Returns:
        numpy.ndarray: Array of preprocessed images.
    """
    data = []  # This list will hold preprocessed images
    skipped_files = []  # To keep track of skipped files

    for idx, filename in enumerate(file_names):
        try:
            # Read image using OpenCV
            img = cv2.imread(os.path.join(directory, filename))

            if img is not None:
                # Convert BGR image to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Resize image to target size
                img = cv2.resize(img, target_size)

                # Normalize pixel values to [0, 1]
                img = img.astype(np.float32) / 255.0

                # Append preprocessed image to data list
                data.append(img)

                # Print the index of the processed image
                print(f"Processed image at index {idx+1}")

            else:
                # If image reading fails, add filename to skipped_files list
                skipped_files.append(filename)
                print(f"Warning: Skipping {filename} due to reading error.")

        except Exception as e:
            # If any other exception occurs during processing, print error
            print(f"Error processing {filename}: {str(e)}")

    if skipped_files:
        print(f"Skipped {len(skipped_files)} files due to errors.")

    # Convert data list to numpy array
    data = np.array(data)

    return data


def custom_range(start, stop, step):
    """
    Generates a custom range of numbers within the specified range.

    Args:
        start (int): Start of the range (inclusive).
        stop (int): End of the range (inclusive).
        step (int): Step size.

    Returns:
        list: List of numbers within the specified range.
    """
    current = start
    result = []

    # Iterate from start to stop (inclusive) with given step
    while current <= stop:
        result.append(current)
        current += step

    return result


def add_gaussian_blur(images, range_set):
    """
    Adds gaussian blur to images.

    Args:
        images (numpy.ndarray): Array of input images.
        range_set (tuple): Range for generating odd numbers for blur kernel size.

    Returns:
        numpy.ndarray: Array of blurred images.
    """
    odd_numbers = custom_range(*range_set)
    blurred_images = []
    for i, image in enumerate(images):
        try:
            ksize = tuple(odd_numbers[i % len(odd_numbers)] for _ in range(2))
            blurred_image = cv2.GaussianBlur(image, ksize, 0)
            blurred_images.append(blurred_image)
        except Exception as e:
            print(f"Error adding blur to image {i}: {str(e)}")
    return np.array(blurred_images)


def add_blur(images, range_set):
    """
    Adds blur to images.

    Args:
        images (numpy.ndarray): Array of input images.
        range_set (tuple): Range for generating odd numbers for blur kernel size.

    Returns:
        numpy.ndarray: Array of blurred images.
    """
    odd_numbers = custom_range(*range_set)
    blurred_images = []
    for i, image in enumerate(images):
        try:
            ksize = tuple(odd_numbers[i % len(odd_numbers)] for _ in range(2))
            blurred_image = cv2.blur(image, ksize)
            blurred_images.append(blurred_image)
        except Exception as e:
            print(f"Error adding blur to image {i}: {str(e)}")
    return np.array(blurred_images)


def display_images(
    arrays,
    labels=None,
    num_samples=10,
    figsize=(20, 5),
    cmap=None,
    is_preprocessed=True,
):
    """
    Displays images from arrays.

    Args:
        arrays (list): List of image arrays.
        labels (list): List of labels for images.
        num_samples (int): Number of samples to display.
        figsize (tuple): Figure size.
        cmap: Color map for images.
        is_preprocessed (bool): Indicates whether images are preprocessed.
    """
    if labels is None:
        labels = [f"Array {i}" for i in range(1, len(arrays) + 1)]
    plt.figure(figsize=figsize)
    for i in range(num_samples):
        for j, array in enumerate(arrays):
            if i < len(array):
                ax = plt.subplot(len(arrays), num_samples, j * num_samples + i + 1)
                image = array[i] if is_preprocessed else plt.imread(array[i])
                plt.imshow(image, cmap=cmap)
                ax.set_title(labels[j])
                ax.axis("off")
    plt.show()


def psnr(y_true, y_pred):
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        y_true (tensor): Ground truth images.
        y_pred (tensor): Predicted images.

    Returns:
        tensor: PSNR value.
    """
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


def ssim(y_true, y_pred):
    """
    Computes Structural Similarity Index (SSIM) between two images.

    Args:
        y_true (tensor): Ground truth images.
        y_pred (tensor): Predicted images.

    Returns:
        tensor: SSIM value.
    """
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


import os


def generate_unique_model_name(model_name, parent_dir):
    """
    Generates a unique model name based on the provided model name and existing models in the parent directory.

    Args:
        model_name (str): Name of the model.
        parent_dir (str): Parent directory for saving models.

    Returns:
        str: Unique model name.
    """
    model_path = os.path.join(parent_dir, "models", f"{model_name}.h5")

    if os.path.exists(model_path):
        # If the model_name already exists, iterate to find the next available number suffix
        suffix = 1
        while True:
            new_model_path = os.path.join(
                parent_dir, "models", f"{model_name}_{suffix}.h5"
            )
            if not os.path.exists(new_model_path):
                return f"{model_name}_{suffix}"
            suffix += 1
    else:
        return model_name


def model_visualization(
    model, parent_dir, model_name, figsize=(15, 15)
):
    # Generate a unique directory name
    unique_model_name = generate_unique_model_name(model_name, parent_dir)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.join(parent_dir, "plots"), exist_ok=True)
    # Generate the visualization and save it as an image file
    font = ImageFont.truetype(
        "arial.ttf", 12
    )  # using comic sans is strictly prohibited!
    visualkeras.layered_view(
        model,
        legend=True,
        font=font,
        to_file=os.path.join(parent_dir, "plots", f"{unique_model_name}.png"),
    )  # font is optional
    # Plot the architecture using plt with little padding
    plt.figure(figsize=figsize)
    img = plt.imread(os.path.join(parent_dir, "plots", f"{unique_model_name}.png"))
    plt.imshow(img)
    plt.axis("off")  # Turn off axis
    plt.show()


def checkpoint_callback(model_name, parent_dir):
    """
    Creates a ModelCheckpoint callback for saving the best model.

    Args:
        model_name (str): Name of the model.
        parent_dir (str): Parent directory for saving models.

    Returns:
        keras.callbacks.ModelCheckpoint: ModelCheckpoint callback.
    """
    # Generate a unique directory name
    unique_model_name = generate_unique_model_name(model_name, parent_dir)
    model_path = os.path.join(parent_dir, "models", f"{unique_model_name}.h5")
    checkpoint = ModelCheckpoint(
        model_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
    )
    return checkpoint, model_path


def tensorboard_callback(parent_dir, model_name):
    """
    Creates a TensorBoard callback for visualization during training.

    Args:
        parent_dir (str): Base directory for storing TensorBoard logs.
        model_name (str): Name of the model.

    Returns:
        tuple: Tuple containing logs directory, experiment directory, and TensorBoard callback.
    """
    unique_model_name = generate_unique_model_name(model_name, parent_dir)
    logs_dir = os.path.join(parent_dir, "logs", unique_model_name)
    os.makedirs(logs_dir, exist_ok=True)  # Create logs directory
    tensorboard_callback = TensorBoard(log_dir=logs_dir, histogram_freq=1)
    return logs_dir, tensorboard_callback


def early_stopping_callback(monitor="val_loss", patience=10):
    """
    Creates an EarlyStopping callback to stop training when validation loss stops improving.

    Returns:
        keras.callbacks.EarlyStopping: EarlyStopping callback.
    """
    return EarlyStopping(monitor=monitor, patience=patience, verbose=1, mode="min")


def create_experiment_notes(**kwargs):
    """
    Creates experiment notes from key-value pairs.

    Args:
        kwargs: Key-value pairs representing experiment details.

    Returns:
        str: Experiment notes in Markdown format.
    """
    experiment_notes = (
        "| Key                 | Value                   |\n"
        "| -------------------- | ----------------------- |\n"
    )

    for key, value in kwargs.items():
        experiment_notes += f"| {key}                 | {value}                   |\n"
    return experiment_notes


def save_image(image, path):
    """
    Saves an image to a specified path.

    Args:
        image (numpy.ndarray): Image data as a NumPy array.
        path (str): Path to save the image.
    """
    # Assuming 'image' is a NumPy array in the range [0, 1]
    image_uint8 = (image * 255).astype(np.uint8)
    Image.fromarray(image_uint8).save(path)


def export_images(images, output_dir):
    """
    Exports images to the specified output directory.

    Args:
        images (numpy.ndarray): Array of images.
        output_dir (str): Directory to save the images.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, image in enumerate(images):
        image_name = f"image_{i+1}.png"
        image_path = os.path.join(output_dir, image_name)
        save_image(image, image_path)
        print(f"Image saved: {image_path}")


def shutdown(time):
    """
    Shutdown Computer
    """
    os.system(f"shutdown /s /t {time}")
