import numpy as np
import cv2
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

cur_path = os.getcwd()

def main():

    # Check command-line arguments
    #if len(sys.argv) not in [2, 3]:
        #sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data('gtsrb')
    print(len(images))
    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE, random_state=42
    )
    print(len(x_train))
    print(len(x_test))
    # Get a compiled neural network
    model = get_model(x_train)

    # Fit model on training data
    model.fit(x_train, y_train,batch_size=32, epochs=EPOCHS)
    #history = model.fit(x_train, y_train, batch_size=32, epochs=EPOCHS, validation_data=(x_test, y_test))

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)



    # Save model to file
    model.save("my_model.h5")
   # if len(sys.argv) == 3:
        #filename = sys.argv[2]
        #model.save(filename)
        #print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    data = []
    labels = []
    # Retrieving the images and their labels
    for i in range(NUM_CATEGORIES):
        path = os.path.join(cur_path, 'gtsrb', str(i))
        images = os.listdir(path)

        for a in images:
            try:
                #image = Image.open(path + '\\' + a)
                image = cv2.imread(path + '\\' + a)
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), 3)
                #image = image.resize((IMG_WIDTH, IMG_HEIGHT))
                image = np.array(image)
                # (width, height , 3)
                #print(image.shape)
                # sim = Image.fromarray(image)
                data.append(image)
                labels.append(i)
            except:
                print("Error loading image")
    # Converting lists into numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    return data,labels

def get_model(X_train):
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Building the model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))

    # Compilation of the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return  model

if __name__ == "__main__":
    main()
