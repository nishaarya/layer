from typing import Any
from layer import Featureset, Train, Dataset
from PIL import Image
import io
import base64
from keras.preprocessing.image import img_to_array
import numpy as np
from keras.callbacks import EarlyStopping
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import Xception
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds


def train_model(train: Train, ds:Dataset("catsdogs"), pf: Featureset("cat_and_dog_features")) -> Any:
    # train: Train, df:Dataset("cats-and-dogs-classification"), pf: Featureset("animal_features")
    """Model train function
    This function is a reserved function and will be called by Layer
    when we want this model to be trained along with the parameters.
    Just like the `features` featureset, you can add more
    parameters to this method to request artifacts (datasets,
    featuresets or models) from Layer.
    Args:
        train (layer.Train): Represents the current train of the model, passed by
            Layer when the training of the model starts.
        pf (spark.DataFrame): Layer will return all features inside the
            `features` featureset as a spark.DataFrame automatically
            joining them by primary keys, described in the dataset.yml
    Returns:
       model: Trained model object
    """
    df = ds.to_pandas().merge(pf.to_pandas(), on='id')
   
    # training and test set 
    training_set = df[(df['path'] == 'training_set/dogs') | (df['path'] == 'training_set/cats')]
    testing_set = df[(df['path'] == 'test_set/dogs') | (df['path'] == 'test_set/cats')]

    X_train = np.stack(training_set['content'].map(load_process_images))
    X_test = np.stack(testing_set['content'].map(load_process_images))

    train.register_input(X_train)
    train.register_output(df['category'])


    load_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1)
    load_datagen.fit(X_train)

    training_data = load_datagen.flow(X_train, training_set['category'], batch_size=32)
    testing_data = load_datagen.flow(X_test, testing_set['category'], batch_size=32)

    
    # Data augmentation
    data_augmentation = keras.Sequential(
        [layers.experimental.preprocessing.RandomFlip("horizontal"), layers.experimental.preprocessing.RandomRotation(0.1),]
    )

    # Base model
    base_model = keras.applications.Xception(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False,
)  # Do not include the ImageNet classifier at the top.

    # Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = keras.Input(shape=(150, 150, 3))
    x = data_augmentation(inputs)  # Apply random data augmentation

    # Pre-trained Xception weights requires that input be scaled
    scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    x = scale_layer(x)

    # Freeze the base model
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

    epochs = 20
    model.fit(training_data, epochs=epochs, validation_data=testing_data)


    # Fine tune
    base_model.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
)

    epochs = 10
    model.fit(training_data, epochs=epochs, validation_data=testing_data)

    test_loss, test_accuracy = model.evaluate(testing_data)
    train_loss, train_accuracy = model.evaluate(training_data)

    train.log_metric("Testing Accuracy", test_accuracy)
    train.log_metric("Testing Loss", test_loss)

    train.log_metric("Training Accuracy", train_accuracy)
    train.log_metric("Training Loss", train_loss)
    return model


def load_process_images(content):
    image_decoded = base64.b64decode(content)
    image = Image.open(io.BytesIO(image_decoded)).resize([150, 150])
    image = img_to_array(image)
    return image
