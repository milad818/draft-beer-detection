#!/usr/bin/env python3

"""Model to classify mugs

This file contains all the model information: the training steps, the batch
size and the model itself.
"""

import os
import cv2
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub


def get_batch_size():
    """Returns the batch size that will be used by your solution.
    It is recommended to change this value.
    """
    return 1

def get_epochs():
    """Returns number of epochs that will be used by your solution.
    It is recommended to change this value.
    """
    return 1

def solution(input_layer):
    """Returns a compiled model.

    This function is expected to return a model to identity the different mugs.
    The model's outputs are expected to be probabilities for the classes and
    and it should be ready for training.
    The input layer specifies the shape of the images. The preprocessing
    applied to the images is specified in data.py.

    Add your solution below.

    Parameters:
        input_layer: An input layer specifying the shape of the input.
            RGB colored images, shape: (width, height, 3)
    Returns:
        model: A compiled model
    """

    BATCH_SIZE = get_batch_size()
    EPOCHS = get_epochs()

    train_dir = "./data/train/"
    eval_dir = "./data/eval/"

    # Check whether all images are of the same shape
    image_shapes = set()

    for root, _, files in os.walk(train_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                image_shapes.add(img.shape[:2])  # (height, width)

    print(f"Unique image shapes: {image_shapes}")

    if len(image_shapes) == 1:
        IMAGE_SHAPE = image_shapes.pop()
        print(f"All images have the same shape: {IMAGE_SHAPE}")
    else:
        print(f"Images have different shapes: {image_shapes}")


    # implement pre-processing and augmentation setting
    # normalization step
    train_datagen = ImageDataGenerator(rescale=1 / 255.)
    train_datagen_augmented = ImageDataGenerator(rescale=1 / 255.,
                                                rotation_range = 20,
                                                shear_range = 0.2,
                                                zoom_range = 0.2,
                                                width_shift_range = 0.2,
                                                height_shift_range = 0.2,
                                                horizontal_flip = True)
    eval_datagen = ImageDataGenerator(rescale=1 / 255.)

    print("Training images:")
    train_data = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(224, 224),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode="categorical",
                                                    seed=42)

    print("Testing images:")
    eval_data = eval_datagen.flow_from_directory(eval_dir,
                                                    target_size=(224, 224),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode="categorical",
                                                    seed=42)

    # UNCOMMENT the few lines below to access class names and print labels
    data_dir = pathlib.Path("./data/train/")  # turn our training path into a Python path
    class_names = np.array(
        sorted([item.name for item in data_dir.glob('*')]))  # created a list of class_names from the subdirectories
    num_class = len(class_names)
    # print(class_names)

    # Define the model using Functional API

    # x = tf.keras.layers.Conv2D(filters=5, kernel_size=3, activation="relu")(input_layer)
    # x = tf.keras.layers.Conv2D(5, 3, activation="relu")(x)
    # x = tf.keras.layers.MaxPool2D(pool_size=2, padding="valid")(x)
    # x = tf.keras.layers.Conv2D(5, 3, activation="relu")(x)
    # x = tf.keras.layers.Conv2D(5, 3, activation="relu")(x)
    # x = tf.keras.layers.MaxPool2D(2)(x)
    # x = tf.keras.layers.Flatten()(x)
    # output_layer = tf.keras.layers.Dense(num_class, activation="softmax")(x)

    ### Modified Version (1) with ADDITIONAL LAYERS to overcome the overfitting issue

    # x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(input_layer)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    # x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    
    # x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    # x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dropout(0.5)(x)  # Helps prevent overfitting
    # x = tf.keras.layers.Dense(128, activation="relu")(x)
    # output_layer = tf.keras.layers.Dense(num_class, activation="softmax")(x)


    ### Modified Version (2) with ADDITIONAL LAYERS to overcome the overfitting issue
    # Convolutional layers with more filters and batch normalization (10 epochs as before)

    # x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation=None, padding="same")(input_layer)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    
    # x = tf.keras.layers.Conv2D(32, 3, activation=None, padding="same")(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    
    # x = tf.keras.layers.MaxPool2D(pool_size=2, padding="valid")(x)
    # x = tf.keras.layers.Dropout(0.25)(x)  # Dropout to reduce overfitting
    
    # # Deeper CNN layers
    # x = tf.keras.layers.Conv2D(64, 3, activation=None, padding="same")(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    
    # x = tf.keras.layers.Conv2D(64, 3, activation=None, padding="same")(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    
    # x = tf.keras.layers.MaxPool2D(2)(x)
    # x = tf.keras.layers.Dropout(0.3)(x)  # Increased dropout
    
    # # Fully connected layers
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(128, activation=None)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    
    # output_layer = tf.keras.layers.Dense(num_class, activation="softmax")(x)


    # DEEPER MODEL 
    # 200 EPOCHS on Google Colab, it is configured automatically stop (it stopped after 140 epochs)
    
    x = tf.keras.layers.Conv2D(32, 3, activation=None, padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.003))(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(32, 3, activation=None, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, padding="valid")(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Block 2 (increased filters to 64)
    x = tf.keras.layers.Conv2D(64, 3, activation=None, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation=None, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Block 3 (added another convolutional block with 128 filters)
    x = tf.keras.layers.Conv2D(128, 3, activation=None, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation=None, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Fully connected layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation=None)(x)  # Increased units to 256
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Output layer
    output_layer = tf.keras.layers.Dense(num_class, activation="softmax")(x)


    # TRANSFER LEARNING implementation
    # Load a pretrained base model (choose ResNet50 or EfficientNet)
    # base_model = hub.KerasLayer(
    #     "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4",  # Change to EfficientNet if needed
    #     trainable=False,  # Keep it frozen
    #     name="feature_extractor",
    #     input_shape=(224, 224) + (3,)
    # )

    # target_shape = (224, 224, 3)
    # resize_layer = tf.keras.layers.Resizing(224, 224)(input_layer)
    # Pass input through base model
    # x = tf.image.resize(input_layer, target_shape[:2])
    # x = base_model(x)

    # Add fully connected layers
    # x = tf.keras.layers.Dense(128, activation="relu")(x)
    # x = tf.keras.layers.Dropout(0.5)(x)  # Prevents overfitting
    # output_layer = tf.keras.layers.Dense(num_class, activation="softmax", name="output")(x)  # 5 classes

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # TODO: Return the compiled model
    return model
