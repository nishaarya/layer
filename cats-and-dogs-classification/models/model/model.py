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

    # simple model

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
  
  # pre-trained

    pretrained_model = Xception(
        weights='imagenet', 
        include_top=False, 
        input_shape=(150, 150, 3))

    # freeze pre-trained model
    pretrained_model.trainable = False

    # model 1 run

    model = models.Sequential()
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(10, activation='sigmoid'))

    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy'])

    model.fit(
        training_data, 
        epochs=20, 
        validation_data=testing_data)

    # Add dense layer

    model = models.Sequential()
    model.add(pretrained_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(10, activation='sigmoid'))

    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy'])

    model.fit(
        training_data, 
        epochs=20, 
        validation_data=testing_data)

    # fine tune

    pretrained_model = Xception(
        weights='imagenet', 
        include_top=False, input_shape=(150, 150, 3))

    pretrained_model.trainable = True
   
   # set_trainable = False

    # for layer in pretrained_model.layers:
    #     if layer == 'block14_sepconv1':
    #         set_trainable = True
            
    #     if set_trainable:
    #         layer.trainable = True
    #     else:
    #         layer.trainable = False

    # train model again

    model = models.Sequential()
    model.add(pretrained_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(10, activation='sigmoid'))
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy'])


    model.fit(
        training_data, 
        epochs=10, 
        validation_data=testing_data)


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
