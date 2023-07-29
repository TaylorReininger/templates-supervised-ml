


# Imports
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np

# Flag to enable suppression of plots
MAKE_PLOTS = False

"""Download the MNIST dataset"""
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


"""Take a look at the data before doing any machine learning"""
# Evaluate the size and shapes of the dataset
print('The data are of the following shapes:')
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Visualize one of the data for sanity checking
if MAKE_PLOTS:
    plt.imshow(x_train[0], cmap='gray')
    plt.title('Image of the Following Digit: {}'.format(y_train[0]))
    plt.show()
    plt.close()

# Make a histogram of the input values to understand the distribution
n_bins = 10
if MAKE_PLOTS:
    plt.hist(x_train.reshape(-1), bins=10)
    plt.grid(color='k', linestyle='-', linewidth=0.5)
    plt.title('Histogram of Training Data')
    plt.show()
    plt.close()

# Do some statistics on the data to see what we're dealing with
mean = np.mean(x_train)
print('Mean of training data: {}'.format(mean))
std = np.std(x_train)
print('Standard deviation of training data: {}'.format(std))
max = np.max(x_train)
print('Maximum value: {}'.format(max))
min = np.min(x_train)
print('Minimum value: {}'.format(min))
unique_classes = np.unique(y_train)
print('Unique classes: {}'.format(unique_classes))
num_classes = len(unique_classes)
print('Number of classes: {}'.format(num_classes))


"""Process the data for machine learning"""
# Normalize the data, intentially only accounting for the training data value range
x_train_norm = (x_train - min)/(max - min)
x_test_norm = (x_test - min)/(max - min)

# Make a histogram of the input values to understand the distribution
n_bins = 10
if MAKE_PLOTS:
    plt.hist(x_train_norm.reshape(-1), bins=n_bins)
    plt.grid(color='k', linestyle='-', linewidth=0.5)
    plt.title('Histogram of Normalized Training Data')
    plt.show()
    plt.close()

# Make a histogram of the output values to ensure a balanced dataset
if MAKE_PLOTS:
    plt.hist(y_train, bins=num_classes)
    plt.grid(color='k', linestyle='-', linewidth=0.5)
    plt.title('Histogram of Output Labels')
    plt.show()
    plt.close()

# Do some statistics on the data to ensure that it was normalized correctly
mean = np.mean(x_train_norm)
print('Mean of normalized training data: {}'.format(mean))
std = np.std(x_train_norm)
print('Standard deviation of normalized training data: {}'.format(std))
max = np.max(x_train_norm)
print('Maximum normalized value: {}'.format(max))
min = np.min(x_train_norm)
print('Minimum normalized value: {}'.format(min))

# Perform a shuffle of the data to avoid training bias 
x_train_norm, y_train = shuffle(x_train_norm, y_train, random_state=0)

# Create one-hot encodings of the label data
y_train_one_hot = tf.one_hot(y_train, num_classes)
y_test_one_hot = tf.one_hot(y_test, num_classes)

# Expand the dimensions of the input data to account for 
x_train_norm = tf.expand_dims(x_train_norm, -1)
x_test_norm = tf.expand_dims(x_test_norm, -1)

print('In shape:')
print(x_train_norm[0].shape)

print('Out shape:')
print(y_train_one_hot[0].shape)

"""Build Convolutional Neural Network (CNN) using the functional syntax for greater flexibility"""
# Create the input layer with the size of the training data
in_tensor = tf.keras.layers.Input(shape=x_train_norm[0].shape, name='input_layer')

# Create the hidden layers with multiple convolution layers, max pooling, and dropout. 
working_tensor = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', name='first_conv_2d')(in_tensor)
working_tensor = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='first_max_pool')(working_tensor)
working_tensor = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='second_conv_2d')(working_tensor)
working_tensor = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='second_max_pool')(working_tensor)
working_tensor = tf.keras.layers.Flatten(name='flatten')(working_tensor)
working_tensor = tf.keras.layers.Dropout(0.5, name='dropout')(working_tensor)

# Create the output layer with the size of the output classes
out_tensor = tf.keras.layers.Dense(num_classes, activation='softmax', name='output_layer')(working_tensor)

# Build the model and display a summary of the model architecture
model = tf.keras.models.Model(inputs=in_tensor, outputs=out_tensor, name='mnist_cnn_model')
model.summary()

"""Train the model"""
# Compile the model for execution
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model on the training data, with a validation split
model.fit(x_train_norm, y_train_one_hot, epochs=5, batch_size=128, validation_split=0.25)








