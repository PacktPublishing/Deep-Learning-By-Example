from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder

import pickle
import tensorflow as tf

cifar10_batches_dir_path = 'cifar-10-batches-py'

tar_gz_filename = 'cifar-10-python.tar.gz'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile(tar_gz_filename):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Python Images Batches') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            tar_gz_filename,
            pbar.hook)

if not isdir(cifar10_batches_dir_path):
    with tarfile.open(tar_gz_filename) as tar:
        tar.extractall()
        tar.close()


# Defining a helper function for loading a batch of images
def load_batch(cifar10_dataset_dir_path, batch_num):
    with open(cifar10_dataset_dir_path + '/data_batch_' + str(batch_num), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    input_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    target_labels = batch['labels']

    return input_features, target_labels


# Defining a function to show the stats for batch ans specific sample
def batch_image_stats(cifar10_dataset_dir_path, batch_num, sample_num):
    batch_nums = list(range(1, 6))

    # checking if the batch_num is a valid batch number
    if batch_num not in batch_nums:
        print('Batch Num is out of Range. You can choose from these Batch nums: {}'.format(batch_nums))
        return None

    input_features, target_labels = load_batch(cifar10_dataset_dir_path, batch_num)

    # checking if the sample_num is a valid sample number
    if not (0 <= sample_num < len(input_features)):
        print('{} samples in batch {}.  {} is not a valid sample number.'.format(len(input_features), batch_num,
                                                                                 sample_num))
        return None

    print('\nStatistics of batch number {}:'.format(batch_num))
    print('Number of samples in this batch: {}'.format(len(input_features)))
    print('Per class counts of each Label: {}'.format(dict(zip(*np.unique(target_labels, return_counts=True)))))

    image = input_features[sample_num]
    label = target_labels[sample_num]
    cifar10_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print('\nSample Image Number {}:'.format(sample_num))
    print('Sample image - Minimum pixel value: {} Maximum pixel value: {}'.format(image.min(), image.max()))
    print('Sample image - Shape: {}'.format(image.shape))
    print('Sample Label - Label Id: {} Name: {}'.format(label, cifar10_class_names[label]))
    plt.axis('off')
    plt.imshow(image)


# Explore a specific batch and sample from the dataset
batch_num = 3
sample_num = 6
batch_image_stats(cifar10_batches_dir_path, batch_num, sample_num)


# Normalize CIFAR-10 images to be in the range of [0,1]

def normalize_images(images):
    # initial zero ndarray
    normalized_images = np.zeros_like(images.astype(float))

    # The first images index is number of images where the other indices indicates
    # hieight, width and depth of the image
    num_images = images.shape[0]

    # Computing the minimum and maximum value of the input image to do the normalization based on them
    maximum_value, minimum_value = images.max(), images.min()

    # Normalize all the pixel values of the images to be from 0 to 1
    for img in range(num_images):
        normalized_images[img, ...] = (images[img, ...] - float(minimum_value)) / float(maximum_value - minimum_value)

    return normalized_images


# encoding the input images. Each image will be represented by a vector of zeros except for the class index of the image
# that this vector represents. The length of this vector depends on number of classes that we have
# the dataset which is 10 in CIFAR-10

def one_hot_encode(images):
    num_classes = 10

    # use sklearn helper function of OneHotEncoder() to do that
    encoder = OneHotEncoder(num_classes)

    # resize the input images to be 2D
    input_images_resized_to_2d = np.array(images).reshape(-1, 1)
    one_hot_encoded_targets = encoder.fit_transform(input_images_resized_to_2d)

    return one_hot_encoded_targets.toarray()


def preprocess_persist_data(cifar10_batches_dir_path, normalize_images, one_hot_encode):
    num_batches = 5
    valid_input_features = []
    valid_target_labels = []

    for batch_ind in range(1, num_batches + 1):
        # Loading batch
        input_features, target_labels = load_batch(cifar10_batches_dir_path, batch_ind)
        num_validation_images = int(len(input_features) * 0.1)

        # Preprocess the current batch and perisist it for future use
        input_features = normalize_images(input_features[:-num_validation_images])
        target_labels = one_hot_encode(target_labels[:-num_validation_images])

        # Persisting the preprocessed batch
        pickle.dump((input_features, target_labels), open('preprocess_train_batch_' + str(batch_ind) + '.p', 'wb'))

        # Define a subset of the training images to be used for validating our model
        valid_input_features.extend(input_features[-num_validation_images:])
        valid_target_labels.extend(target_labels[-num_validation_images:])

    # Preprocessing and persisting the validationi subset
    input_features = normalize_images(np.array(valid_input_features))
    target_labels = one_hot_encode(np.array(valid_target_labels))

    pickle.dump((input_features, target_labels), open('preprocess_valid.p', 'wb'))

    # Now it's time to preporcess and persist the test batche
    with open(cifar10_batches_dir_path + '/test_batch', mode='rb') as file:
        test_batch = pickle.load(file, encoding='latin1')

    test_input_features = test_batch['data'].reshape((len(test_batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_input_labels = test_batch['labels']

    # Normalizing and encoding the test batch
    input_features = normalize_images(np.array(test_input_features))
    target_labels = one_hot_encode(np.array(test_input_labels))

    pickle.dump((input_features, target_labels), open('preprocess_test.p', 'wb'))


# Calling the helper function above to preprocess and persist the training, validation, and testing set
preprocess_persist_data(cifar10_batches_dir_path, normalize_images, one_hot_encode)

# Load the Preprocessed Validation data
valid_input_features, valid_input_labels = pickle.load(open('preprocess_valid.p', mode='rb'))


# Defining the model inputs
def images_input(img_shape):
    return tf.placeholder(tf.float32, (None,) + img_shape, name="input_images")


def target_input(num_classes):
    target_input = tf.placeholder(tf.int32, (None, num_classes), name="input_images_target")
    return target_input


# define a function for the dropout layer keep probability
def keep_prob_input():
    return tf.placeholder(tf.float32, name="keep_prob")


tf.reset_default_graph()


# Applying a convolution operation to the input tensor followed by max pooling
def conv2d_layer(input_tensor, conv_layer_num_outputs, conv_kernel_size, conv_layer_strides, pool_kernel_size,
                 pool_layer_strides):
    input_depth = input_tensor.get_shape()[3].value
    weight_shape = conv_kernel_size + (input_depth, conv_layer_num_outputs,)

    # Defining layer weights and biases
    weights = tf.Variable(tf.random_normal(weight_shape))
    biases = tf.Variable(tf.random_normal((conv_layer_num_outputs,)))

    # Considering the biase variable
    conv_strides = (1,) + conv_layer_strides + (1,)

    conv_layer = tf.nn.conv2d(input_tensor, weights, strides=conv_strides, padding='SAME')
    conv_layer = tf.nn.bias_add(conv_layer, biases)

    conv_kernel_size = (1,) + conv_kernel_size + (1,)

    pool_strides = (1,) + pool_layer_strides + (1,)

    pool_layer = tf.nn.max_pool(conv_layer, ksize=conv_kernel_size, strides=pool_strides, padding='SAME')

    return pool_layer

#Flatten the output of max pooling layer to be fing to the fully connected layer which only accepts the output
# to be in 2D
def flatten_layer(input_tensor):

    return tf.contrib.layers.flatten(input_tensor)

#Define the fully connected layer that will use the flattened output of the stacked convolution layers
#to do the actuall classification
def fully_connected_layer(input_tensor, num_outputs):
    return tf.layers.dense(input_tensor, num_outputs)

#Defining the output function
def output_layer(input_tensor, num_outputs):
    return  tf.layers.dense(input_tensor, num_outputs)


def build_convolution_net(image_data, keep_prob):
    # Applying 3 convolution layers followed by max pooling layers
    conv_layer_1 = conv2d_layer(image_data, 32, (3, 3), (1, 1), (3, 3), (3, 3))
    conv_layer_2 = conv2d_layer(conv_layer_1, 64, (3, 3), (1, 1), (3, 3), (3, 3))
    conv_layer_3 = conv2d_layer(conv_layer_2, 128, (3, 3), (1, 1), (3, 3), (3, 3))

    # Flatten the output from 4D to 2D to be fed to the fully connected layer
    flatten_output = flatten_layer(conv_layer_3)

    # Applying 2 fully connected layers with drop out
    fully_connected_layer_1 = fully_connected_layer(flatten_output, 64)
    fully_connected_layer_1 = tf.nn.dropout(fully_connected_layer_1, keep_prob)
    fully_connected_layer_2 = fully_connected_layer(fully_connected_layer_1, 32)
    fully_connected_layer_2 = tf.nn.dropout(fully_connected_layer_2, keep_prob)

    # Applying the output layer while the output size will be the number of categories that we have
    # in CIFAR-10 dataset
    output_logits = output_layer(fully_connected_layer_2, 10)

    # returning output
    return output_logits
#Using the helper function above to build the network

#First off, let's remove all the previous inputs, weights, biases form the previous runs
tf.reset_default_graph()

# Defining the input placeholders to the convolution neural network
input_images = images_input((32, 32, 3))
input_images_target = target_input(10)
keep_prob = keep_prob_input()

# Building the models
logits_values = build_convolution_net(input_images, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits_values = tf.identity(logits_values, name='logits')

# defining the model loss
model_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_values, labels=input_images_target))

# Defining the model optimizer
model_optimizer = tf.train.AdamOptimizer().minimize(model_cost)

# Calculating and averaging the model accuracy
correct_prediction = tf.equal(tf.argmax(logits_values, 1), tf.argmax(input_images_target, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='model_accuracy')

#Define a helper function for kicking off the training process
def train(session, model_optimizer, keep_probability, in_feature_batch, target_batch):

    session.run(model_optimizer, feed_dict={input_images: in_feature_batch, input_images_target: target_batch, keep_prob: keep_probability})


# Defining a helper funcitno for print information about the model accuracy and it's validation accuracy as well
def print_model_stats(session, input_feature_batch, target_label_batch, model_cost, model_accuracy):
    validation_loss = session.run(model_cost,
                                  feed_dict={input_images: input_feature_batch, input_images_target: target_label_batch,
                                             keep_prob: 1.0})
    validation_accuracy = session.run(model_accuracy, feed_dict={input_images: input_feature_batch,
                                                                 input_images_target: target_label_batch,
                                                                 keep_prob: 1.0})

    print("Valid Loss: %f" % (validation_loss))
    print("Valid accuracy: %f" % (validation_accuracy))

# Model Hyperparameters
num_epochs = 100
batch_size = 128
keep_probability = 0.5

# Splitting the dataset features and labels to batches
def batch_split_features_labels(input_features, target_labels, train_batch_size):
    for start in range(0, len(input_features), train_batch_size):
        end = min(start + train_batch_size, len(input_features))
        yield input_features[start:end], target_labels[start:end]

#Loading the persisted preprocessed training batches
def load_preprocess_training_batch(batch_id, batch_size):
    filename = 'preprocess_train_batch_' + str(batch_id) + '.p'
    input_features, target_labels = pickle.load(open(filename, mode='rb'))

    # Returning the training images in batches according to the batch size defined above
    return batch_split_features_labels(input_features, target_labels, train_batch_size)


print('Training on only a Single Batch from the CIFAR-10 Dataset...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(num_epochs):
        batch_ind = 1

        for batch_features, batch_labels in load_preprocess_training_batch(batch_ind, batch_size):
            train(sess, model_optimizer, keep_probability, batch_features, batch_labels)

        print('Epoch number {:>2}, CIFAR-10 Batch Number {}:  '.format(epoch + 1, batch_ind), end='')
        print_model_stats(sess, batch_features, batch_labels, model_cost, accuracy)

print('Full training for the network...')
model_save_path = './cifar-10_classification'

with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(num_epochs):

        # iterate through the batches
        num_batches = 5

        for batch_ind in range(1, num_batches + 1):
            for batch_features, batch_labels in load_preprocess_training_batch(batch_ind, batch_size):
                train(sess, model_optimizer, keep_probability, batch_features, batch_labels)

            print('Epoch number{:>2}, CIFAR-10 Batch Number {}:  '.format(epoch + 1, batch_ind), end='')
            print_model_stats(sess, batch_features, batch_labels, model_cost, accuracy)

    # Save the trained Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, model_save_path)


# A helper function to visualize some samples and their corresponding predictions
def display_samples_predictions(input_features, target_labels, samples_predictions):
    num_classes = 10

    cifar10_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(num_classes))
    label_inds = label_binarizer.inverse_transform(np.array(target_labels))

    fig, axies = plt.subplots(nrows=4, ncols=2)
    fig.tight_layout()
    fig.suptitle('Softmax Predictions', fontsize=20, y=1.1)

    num_predictions = 4
    margin = 0.05
    ind = np.arange(num_predictions)
    width = (1. - 2. * margin) / num_predictions

    for image_ind, (feature, label_ind, prediction_indicies, prediction_values) in enumerate(
            zip(input_features, label_inds, samples_predictions.indices, samples_predictions.values)):
        prediction_names = [cifar10_class_names[pred_i] for pred_i in prediction_indicies]
        correct_name = cifar10_class_names[label_ind]

        axies[image_ind][0].imshow(feature)
        axies[image_ind][0].set_title(correct_name)
        axies[image_ind][0].set_axis_off()

        axies[image_ind][1].barh(ind + margin, prediction_values[::-1], width)
        axies[image_ind][1].set_yticks(ind + margin)
        axies[image_ind][1].set_yticklabels(prediction_names[::-1])
        axies[image_ind][1].set_xticks([0, 0.5, 1.0])


test_batch_size = 64

save_model_path = './cifar-10_classification'

# Number of images to visualize
num_samples = 4

# Number of top predictions
top_n_predictions = 4


# Defining a helper function for testing the trained model
def test_classification_model():
    input_test_features, target_test_labels = pickle.load(open('preprocess_test.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # loading the trained model
        model = tf.train.import_meta_graph(save_model_path + '.meta')
        model.restore(sess, save_model_path)

        # Getting some input and output Tensors from loaded model
        model_input_values = loaded_graph.get_tensor_by_name('input_images:0')
        model_target = loaded_graph.get_tensor_by_name('input_images_target:0')
        model_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        model_logits = loaded_graph.get_tensor_by_name('logits:0')
        model_accuracy = loaded_graph.get_tensor_by_name('model_accuracy:0')

        # Testing the trained model on the test set batches
        test_batch_accuracy_total = 0
        test_batch_count = 0

        for input_test_feature_batch, input_test_label_batch in batch_split_features_labels(input_test_features,
                                                                                            target_test_labels,
                                                                                            test_batch_size):
            test_batch_accuracy_total += sess.run(
                model_accuracy,
                feed_dict={model_input_values: input_test_feature_batch, model_target: input_test_label_batch,
                           model_keep_prob: 1.0})
            test_batch_count += 1

        print('Test set accuracy: {}\n'.format(test_batch_accuracy_total / test_batch_count))

        # print some random images and their corresponding predictions from the test set results
        random_input_test_features, random_test_target_labels = tuple(
            zip(*random.sample(list(zip(input_test_features, target_test_labels)), num_samples)))

        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(model_logits), top_n_predictions),
            feed_dict={model_input_values: random_input_test_features, model_target: random_test_target_labels,
                       model_keep_prob: 1.0})

        display_samples_predictions(random_input_test_features, random_test_target_labels, random_test_predictions)


# Calling the function
test_classification_model()