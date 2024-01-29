
# Imports
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import pickle
import os

# Flag to enable suppression of plots
MAKE_PLOTS = True
DIR_SAVE_NETS = 'trained_nets'

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
epochs = 5
history = model.fit(x_train_norm, y_train_one_hot, epochs=epochs, batch_size=128, validation_split=0.25)

if not os.path.exists(DIR_SAVE_NETS):
    os.makedirs(DIR_SAVE_NETS)

# Save the model to disk
path_trained_model = os.path.join(DIR_SAVE_NETS, 'trained_model.keras')
model.save(path_trained_model)

# Save the history to disk
path_model_history = os.path.join(DIR_SAVE_NETS, 'model_history')
with open(path_model_history, 'wb') as history_file:
    pickle.dump(history.history, history_file)

# Load the model and history from disk
new_model = tf.keras.models.load_model(path_trained_model)
with open(path_model_history, "rb") as history_file:
    new_history = pickle.load(history_file)


# Make a plot of the training and validation accuracies and losses
if MAKE_PLOTS:
    plt.plot(new_history['accuracy'])
    plt.plot(new_history['val_accuracy'])
    plt.title('Training and Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.xlim([0, epochs-1]) 
    plt.legend(['train', 'val'], loc='lower right')
    plt.grid(color='k', linestyle='-', linewidth=0.5)
    plt.show()
    plt.close()
    
    plt.plot(new_history['loss'])
    plt.plot(new_history['val_loss'])
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlim([0, epochs-1])
    plt.legend(['train', 'val'], loc='upper right')
    plt.grid(color='k', linestyle='-', linewidth=0.5)
    plt.show()
    plt.close()


"""Do Testing on the Results"""
# Perform inference with the holdout x data
test_y_hat = new_model.predict(x_test_norm)

# Produce a confusion matrix
confusion = confusion_matrix(y_test, np.argmax(test_y_hat, axis=1))
# Normalize the confusion matrix
conf_norm = np.zeros(confusion.shape)
for row in range(0, len(unique_classes)):
    conf_norm[row, :] = confusion[row, :]/np.sum(confusion[row, :])

# Plot the confusion matrix with seaborn
if MAKE_PLOTS:
    sns.heatmap(conf_norm, annot=True, fmt='0.2f', cmap=sns.color_palette("Blues", as_cmap=True),
                xticklabels=unique_classes, yticklabels=unique_classes, vmin=0, vmax=1)
    plt.ylabel('Prediction')
    plt.xlabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Get the overall accuracy and loss and display them to the user
evaluation = new_model.evaluate(x_test_norm, y_test_one_hot)
print('------------------------------------')
print('The final test accuracy is: %0.4f' % (evaluation[1]))
print('The final test loss is: %0.3e' % (evaluation[0]))






