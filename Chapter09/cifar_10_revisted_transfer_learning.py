import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os

# Importing a helper module for the functions of the Inception model.
import inception

import cifar10
from cifar10 import num_classes

from inception import transfer_values_cache

#Importing the color map for plotting each class with different color.
import matplotlib.cm as color_map


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix



cifar10.data_path = "data/CIFAR-10/"
cifar10.maybe_download_and_extract()

class_names = cifar10.load_class_names()
print(class_names)

print('Loading the training set...')
training_images, training_cls_integers, trainig_one_hot_labels = cifar10.load_training_data()

print('Loading the test set...')
testing_images, testing_cls_integers, testing_one_hot_labels = cifar10.load_test_data()

print("-Number of images in the training set:\t\t{}".format(len(training_images)))
print("-Number of images in the testing set:\t\t{}".format(len(testing_images)))


def plot_imgs(imgs, true_class, predicted_class=None):
    assert len(imgs) == len(true_class)

    # Creating a placeholders for 9 subplots
    fig, axes = plt.subplots(3, 3)

    # Adjustting spacing.
    if predicted_class is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(imgs):
            # Plot image.
            ax.imshow(imgs[i],
                      interpolation='nearest')

            # Get the actual name of the true class from the class_names array
            true_class_name = class_names[true_class[i]]

            # Showing labels for the predicted and true classes
            if predicted_class is None:
                xlabel = "True: {0}".format(true_class_name)
            else:
                # Name of the predicted class.
                predicted_class_name = class_names[predicted_class[i]]

                xlabel = "True: {0}\nPred: {1}".format(true_class_name, predicted_class_name)

            ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

# get the first 9 images in the test set
imgs = testing_images[0:9]

# Get the integer representation of the true class.
true_class = testing_cls_integers[0:9]

# Plotting the images
plot_imgs(imgs=imgs, true_class=true_class)

print('Downloading the pretrained inception v3 model')
inception.maybe_download()

# Loading the inception model so that we can inialized it with the pretrained weights and customize for our model
inception_model = inception.Inception()

file_path_train = os.path.join(cifar10.data_path, 'inception_cifar10_train.pkl')
file_path_test = os.path.join(cifar10.data_path, 'inception_cifar10_test.pkl')

print("Processing Inception transfer-values for the training images of Cifar-10 ...")

# First we need to scale the imgs to fit the Inception model requirements as it requires all pixels to be from 0 to 255,
# while our training examples of the CIFAR-10 pixels are between 0.0 and 1.0
imgs_scaled = training_images * 255.0

# Checking if the transfer-values for our training images are already calculated and loading them, if not calcaulate and save them.
transfer_values_training = transfer_values_cache(cache_path=file_path_train,
                                              images=imgs_scaled,
                                              model=inception_model)

print("Processing Inception transfer-values for the testing images of Cifar-10 ...")

# First we need to scale the imgs to fit the Inception model requirements as it requires all pixels to be from 0 to 255,
# while our training examples of the CIFAR-10 pixels are between 0.0 and 1.0
imgs_scaled = testing_images * 255.0

# Checking if the transfer-values for our training images are already calculated and loading them, if not calcaulate and save them.
transfer_values_testing = transfer_values_cache(cache_path=file_path_test,
                                             images=imgs_scaled,
                                             model=inception_model)

print('Shape of the training set transfer values...')
print(transfer_values_training.shape)

print('Shape of the testing set transfer values...')
print(transfer_values_testing.shape)


def plot_transferValues(ind):
    print("Original input image:")

    # Plot the image at index ind of the test set.
    plt.imshow(testing_images[ind], interpolation='nearest')
    plt.show()

    print("Transfer values using Inception model:")

    # Visualize the transfer values as an image.
    transferValues_img = transfer_values_testing[ind]
    transferValues_img = transferValues_img.reshape((32, 64))

    # Plotting the transfer values image.
    plt.imshow(transferValues_img, interpolation='nearest', cmap='Reds')
    plt.show()

plot_transferValues(ind=15)

pca_obj = PCA(n_components=2)
subset_transferValues = transfer_values_training[0:3000]

cls_integers = testing_cls_integers[0:3000]

print('Shape of a subset form the transfer values...')
print(subset_transferValues.shape)

reduced_transferValues = pca_obj.fit_transform(subset_transferValues)
print('Shape of the reduced version of the transfer values...')
print(reduced_transferValues.shape)


def plot_reduced_transferValues(transferValues, cls_integers):
    # Create a color-map with a different color for each class.
    c_map = color_map.rainbow(np.linspace(0.0, 1.0, num_classes))

    # Getting the color for each sample.
    colors = c_map[cls_integers]

    # Getting the x and y values.
    x_val = transferValues[:, 0]
    y_val = transferValues[:, 1]

    # Plot the transfer values in a scatter plot
    plt.scatter(x_val, y_val, color=colors)
    plt.show()


plot_reduced_transferValues(reduced_transferValues, cls_integers)

pca_obj = PCA(n_components=50)
transferValues_50d = pca_obj.fit_transform(subset_transferValues)
tsne_obj = TSNE(n_components=2)

reduced_transferValues = tsne_obj.fit_transform(transferValues_50d)

print('Shape of the reduced version of the transfer values using t-SNE method...')
print(reduced_transferValues.shape)

plot_reduced_transferValues(reduced_transferValues, cls_integers)

transferValues_arrLength = inception_model.transfer_len
input_values = tf.placeholder(tf.float32, shape=[None, transferValues_arrLength], name='input_values')
y_actual = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_actual')
y_actual_cls = tf.argmax(y_actual, axis=1)

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

# First fully-connected layer.
layer_fc1 = new_fc_layer(input=input_values,
                             num_inputs=2048,
                             num_outputs=1024,
                             use_relu=True)

# Second fully-connected layer.
layer_fc2 = new_fc_layer(input=layer_fc1,
                             num_inputs=1024,
                             num_outputs=num_classes,
                             use_relu=False)

# Predicted class-label.
y_predicted = tf.nn.softmax(layer_fc2)

# Cross-entropy for the classification of each image.
cross_entropy = \
    tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                labels=y_actual)

# Loss aka. cost-measure.
# This is the scalar value that must be minimized.
loss = tf.reduce_mean(cross_entropy)

step = tf.Variable(initial_value=0,
                          name='step', trainable=False)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, step)

y_predicted_cls = tf.argmax(y_predicted, axis=1)
correct_prediction = tf.equal(y_predicted_cls, y_actual_cls)

model_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

training_batch_size = 32

def select_random_batch():
    # Number of images (transfer-values) in the training-set.
    num_imgs = len(transfer_values_training)

    # Create a random index.
    ind = np.random.choice(num_imgs,
                           size=training_batch_size,
                           replace=False)

    # Use the random index to select random x and y-values.
    # We use the transfer-values instead of images as x-values.
    x_batch = transfer_values_training[ind]
    y_batch = trainig_one_hot_labels[ind]

    return x_batch, y_batch


def optimize(num_iterations):
    for i in range(num_iterations):
        # Selectin a random batch of images for training
        # where the transfer values of the images will be stored in input_batch
        # and the actual labels of those batch of images will be stored in y_actual_batch
        input_batch, y_actual_batch = select_random_batch()

        # storing the batch in a dict with the proper names
        # such as the input placeholder variables that we define above.
        feed_dict = {input_values: input_batch,
                     y_actual: y_actual_batch}

        # Now we call the optimizer of this batch of images
        # TensorFlow will automatically feed the values of the dict we created above
        # to the model input placeholder variables that we defined above.
        i_global, _ = session.run([step, optimizer],
                                  feed_dict=feed_dict)

        # print the accuracy every 100 steps.
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-batch.
            batch_accuracy = session.run(model_accuracy,
                                         feed_dict=feed_dict)

            msg = "Step: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_accuracy))


def plot_errors(cls_predicted, cls_correct):
    # cls_predicted is an array of the predicted class-number for
    # all images in the test-set.

    # cls_correct is an array with boolean values to indicate
    # whether is the model predicted the correct class or not.

    # Negate the boolean array.
    incorrect = (cls_correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.

    incorrectly_classified_images = testing_images[incorrect]

    # Get the predicted classes for those images.
    cls_predicted = cls_predicted[incorrect]

    # Get the true classes for those images.
    true_class = testing_cls_integers[incorrect]

    n = min(9, len(incorrectly_classified_images))

    # Plot the first n images.
    plot_imgs(imgs=incorrectly_classified_images[0:n],
              true_class=true_class[0:n],
              predicted_class=cls_predicted[0:n])


def plot_confusionMatrix(cls_predicted):

    # cls_predicted array of all the predicted
    # classes numbers in the test.

    # Call the confucion matrix of sklearn
    cm = confusion_matrix(y_true=testing_cls_integers,
                          y_pred=cls_predicted)

    # Printing the confusion matrix
    for i in range(num_classes):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    # labeling each column of the confusion matrix with the class number
    cls_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(cls_numbers))


# Split the data-set in batches of this size to limit RAM usage.
batch_size = 128


def predict_class(transferValues, labels, cls_true):
    # Number of images.
    num_imgs = len(transferValues)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_predicted = np.zeros(shape=num_imgs, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_imgs:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_imgs)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {input_values: transferValues[i:j],
                     y_actual: labels[i:j]}

        # Calculate the predicted class using TensorFlow.
        cls_predicted[i:j] = session.run(y_predicted_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = [a == p for a, p in zip(cls_true, cls_predicted)]


    print(type(correct))

    return correct, cls_predicted

def predict_class_test():
    return predict_class(transferValues = transfer_values_testing,
                       labels = trainig_one_hot_labels,
                       cls_true = training_cls_integers)

def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True means 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.

    # Return the classification accuracy
    # and the number of correct classifications.
    return np.mean(correct), np.sum(correct)


def test_accuracy(show_example_errors=False,
                  show_confusion_matrix=False):
    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_class_test()

    print(type(correct))

    # Classification accuracypredict_class_test and the number of correct classifications.
    accuracy, num_correct = classification_accuracy(correct)

    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Test set accuracy: {0:.1%} ({1} / {2})"
    print(msg.format(accuracy, num_correct, num_images))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_errors(cls_predicted=cls_pred, cls_correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusionMatrix(cls_predicted=cls_pred)

test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

optimize(num_iterations=1000)

test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)