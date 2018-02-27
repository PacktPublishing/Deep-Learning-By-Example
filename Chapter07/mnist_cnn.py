#importing required packages
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

#Defining the hyper parameters of the 2 convolution layers and fully connected layer

# first convolution layer
filter_size_1 = 5          # the size fo the feature detector will be 5 by 5.
filters_1 = 16         # Using 16 filters.

# second convolution layer
filter_size_2 = 5
filters_2 = 36

# FC layer
fc_num_neurons = 128 # number of neurons in the fully connected layer


#loading the mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('data/MNIST/', one_hot=True)

print("- Number of images in the training set:\t\t{}".format(len(mnist_data.train.labels)))
print("- Number of images in the test set:\t\t{}".format(len(mnist_data.test.labels)))
print("- Number of images in the validation set:\t{}".format(len(mnist_data.validation.labels)))

mnist_data.test.cls_integer = np.argmax(mnist_data.test.labels, axis=1)

# Default size for the input monocrome images of MNIST
image_size = 28

# Each image is stored as vector of this size.
image_size_flat = image_size * image_size

# The shape of each image
image_shape = (image_size, image_size)

# All the images in the mnist dataset are stored as a monocrome with only 1 channel
num_channels = 1

# Number of classes in the MNIST dataset from 0 till 9 which is 10
num_classes = 10


def plot_imgs(imgs, cls_actual, cls_predicted=None):
    assert len(imgs) == len(cls_actual) == 9

    # create a figure with 9 subplots to plot the images.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # plot the image at the ith index
        ax.imshow(imgs[i].reshape(image_shape), cmap='binary')

        # labeling the images with the actual and predicted classes.
        if cls_predicted is None:
            xlabel = "True: {0}".format(cls_actual[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_actual[i], cls_predicted[i])

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

    plt.show()

# Visualizing 9 images form the test set.
imgs = mnist_data.test.images[0:9]

# getting the actual classes of these 9 images
cls_actual = mnist_data.test.cls_integer[0:9]

#plotting the images
plot_imgs(imgs=imgs, cls_actual=cls_actual)

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def conv_layer(input,  # the output of the previous layer.
               input_channels,
               filter_size,
               filters,
               use_pooling=True):  # Use 2x2 max-pooling.

    # preparing the accepted shape of the input Tensor.
    shape = [filter_size, filter_size, input_channels, filters]

    # Create weights which means filters with the given shape.
    filters_weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    filters_biases = new_biases(length=filters)

    # Calling the conve2d function as we explained above, were the strides parameter
    # has four values the first one for the image number and the last 1 for the input image channel
    # the middle ones represents how many pixels the filter should move with in the x and y axis
    conv_layer = tf.nn.conv2d(input=input,
                              filter=filters_weights,
                              strides=[1, 1, 1, 1],
                              padding='SAME')

    # Adding the biase to the output of the conv_layer.
    conv_layer += filters_biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # reduce the output feature map by max_pool layer
        pool_layer = tf.nn.max_pool(value=conv_layer,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')

    # feeding the output to a ReLU activation function.
    relu_layer = tf.nn.relu(pool_layer)

    # return the final results after applying relu and the filter weights
    return relu_layer, filters_weights


def flatten_layer(layer):
    # Get the shape of layer.
    shape = layer.get_shape()

    # We need to flatten the layer which has the shape of The shape  [num_images, image_height, image_width, num_channels]
    # so that it has the shape of [batch_size, num_features] where number_features is image_height * image_width * num_channels

    number_features = shape[1:4].num_elements()

    # Reshaping that to be fed to the fully connected layer
    flatten_layer = tf.reshape(layer, [-1, number_features])

    # Return both the flattened layer and the number of features.
    return flatten_layer, number_features

def fc_layer(input,          # the flatten output.
                 num_inputs,     # Number of inputs from previous layer
                 num_outputs,    # Number of outputs
                 use_relu=True): # Use ReLU on the output to remove negative values

    # Creating the weights for the neurons of this fc_layer
    fc_weights = new_weights(shape=[num_inputs, num_outputs])
    fc_biases = new_biases(length=num_outputs)

    # Calculate the layer values by doing matrix multiplication of
    # the input values and fc_weights, and then add the fc_bias-values.
    fc_layer = tf.matmul(input, fc_weights) + fc_biases

    # if use RelU parameter is true
    if use_relu:
        relu_layer = tf.nn.relu(fc_layer)
        return relu_layer

    return fc_layer

input_values = tf.placeholder(tf.float32, shape=[None, image_size_flat], name='input_values')

input_image = tf.reshape(input_values, [-1, image_size, image_size, num_channels])

y_actual = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

y_actual_cls_integer = tf.argmax(y_actual, axis=1)

#Building the model
conv_layer_1, conv1_weights = \
        conv_layer(input=input_image,
                   input_channels=num_channels,
                   filter_size=filter_size_1,
                   filters=filters_1,
                   use_pooling=True)

conv_layer_2, conv2_weights = \
         conv_layer(input=conv_layer_1,
                   input_channels=filters_1,
                   filter_size=filter_size_2,
                   filters=filters_2,
                   use_pooling=True)

flatten_layer, number_features = flatten_layer(conv_layer_2)

fc_layer_1 = fc_layer(input=flatten_layer,
                         num_inputs=number_features,
                         num_outputs=fc_num_neurons,
                         use_relu=True)

fc_layer_2 = fc_layer(input=fc_layer_1,
                         num_inputs=fc_num_neurons,
                         num_outputs=num_classes,
                         use_relu=False)

y_predicted = tf.nn.softmax(fc_layer_2)

y_predicted_cls_integer = tf.argmax(y_predicted, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc_layer_2,
                                                        labels=y_actual)
# Getting the model averaged cost
model_cost = tf.reduce_mean(cross_entropy)

# Defining the model optimizer
model_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(model_cost)

model_correct_prediction = tf.equal(y_predicted_cls_integer, y_actual_cls_integer)
model_accuracy = tf.reduce_mean(tf.cast(model_correct_prediction, tf.float32))

# Training the model
session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = 64

# number of optimization iterations performed so far
total_iterations = 0


def optimize(num_iterations):
    # Update globally the total number of iterations performed so far.
    global total_iterations

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Generating a random batch for the training process
        # input_batch now contains a bunch of images from the training set and
        # y_actual_batch are the actual labels for the images in the input batch.
        input_batch, y_actual_batch = mnist_data.train.next_batch(train_batch_size)

        # Putting the previous values in a dict format for Tensorflow to automatically assign them to the input
        # placeholders that we defined above
        feed_dict = {input_values: input_batch,
                     y_actual: y_actual_batch}

        # Next up, we run the model optimizer on this batch of images
        session.run(model_optimizer, feed_dict=feed_dict)

        # Print the training status every 100 iterations.
        if i % 100 == 0:
            # measuring the accuracy over the training set.
            acc_training_set = session.run(model_accuracy, feed_dict=feed_dict)

            # Printing the accuracy over the training set
            print("Iteration: {0:>6}, Accuracy Over the training set: {1:>6.1%}".format(i + 1, acc_training_set))

    # Update the number of iterations performed so far
    total_iterations += num_iterations


def plot_errors(cls_predicted, correct):
    # cls_predicted is an array of the predicted class number of each image in the test set.


    # Extracting the incorrect images.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = mnist_data.test.images[incorrect]

    # Get the predicted classes for those incorrect images.
    cls_pred = cls_predicted[incorrect]

    # Get the actual classes for those incorrect images.
    cls_true = mnist_data.test.cls_integer[incorrect]

    # Plot 9 of these images
    plot_imgs(imgs=imgs[0:9],
              cls_actual=cls_actual[0:9],
              cls_predicted=cls_predicted[0:9])


def plot_confusionMatrix(cls_predicted):
    # cls_predicted is an array of the predicted class number of each image in the test set.

    # Get the actual classes for the test-set.
    cls_actual = mnist_data.test.cls_integer

    # Generate the confusion matrix using sklearn.
    conf_matrix = confusion_matrix(y_true=cls_actual,
                                   y_pred=cls_predicted)

    # Print the matrix.
    print(conf_matrix)

    # visualizing the confusion matrix.
    plt.matshow(conf_matrix)

    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted class')
    plt.ylabel('True class')

    # Showing the plot
    plt.show()


# measuring the accuracy of the trained model over the test set by splitting it into small batches
test_batch_size = 256


def test_accuracy(show_errors=False,
                  show_confusionMatrix=False):
    # number of test images
    number_test = len(mnist_data.test.images)

    # define an array of zeros for the predicted classes of the test set which
    # will be measured in mini batches and stored it.
    cls_predicted = np.zeros(shape=number_test, dtype=np.int)

    # measuring the predicted classes for the testing batches.

    # Starting by the batch at index 0.
    i = 0

    while i < number_test:
        # The ending index for the next batch to be processed is j.
        j = min(i + test_batch_size, number_test)

        # Getting all the images form the test set between the start and end indices
        input_images = mnist_data.test.images[i:j, :]

        # Get the acutal labels for those images.
        actual_labels = mnist_data.test.labels[i:j, :]

        # Create a feed-dict with the corresponding values for the input placeholder values
        feed_dict = {input_values: input_images,
                     y_actual: actual_labels}

        cls_predicted[i:j] = session.run(y_predicted_cls_integer, feed_dict=feed_dict)

        # Setting the start of the next batch to be the end of the one that we just processed j
        i = j

    # Get the actual class numbers of the test images.
    cls_actual = mnist_data.test.cls_integer

    # Check if the model predictions are correct or not
    correct = (cls_actual == cls_predicted)

    # Summing up the correct examples
    correct_number_images = correct.sum()

    # measuring the accuracy by dividing the correclty classified ones with total number of images in the test set.
    testset_accuracy = float(correct_number_images) / number_test

    # showing the accuracy.
    print("Accuracy on Test-Set: {0:.1%} ({1} / {2})".format(testset_accuracy, correct_number_images, number_test))

    # showing some examples form the incorrect ones.
    if show_errors:
        print("Example errors:")
        plot_errors(cls_predicted=cls_predicted, correct=correct)

    # Showing the confusion matrix of the test set predictions
    if show_confusionMatrix:
        print("Confusion Matrix:")
        plot_confusionMatrix(cls_predicted=cls_predicted)


# Start the training
optimize(num_iterations=10000)

test_accuracy(show_errors=True,
                    show_confusionMatrix=True)



