import matplotlib.pyplot as plt
import pickle as pkl

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist_dataset = input_data.read_data_sets('MNIST_data')


# Defining the model input for the generator and discrimator
def inputs_placeholders(discrimator_real_dim, gen_z_dim):
    real_discrminator_input = tf.placeholder(tf.float32, (None, discrimator_real_dim), name="real_discrminator_input")
    generator_inputs_z = tf.placeholder(tf.float32, (None, gen_z_dim), name="generator_input_z")

    return real_discrminator_input, generator_inputs_z


def generator(gen_z, gen_out_dim, num_hiddern_units=128, reuse_vars=False, leaky_relu_alpha=0.01):
    ''' Building the generator part of the network

        Function arguments
        ---------
        gen_z : the generator input tensor
        gen_out_dim : the output shape of the generator
        num_hiddern_units : Number of neurons/units in the hidden layer
        reuse_vars : Reuse variables with tf.variable_scope
        leaky_relu_alpha : leaky ReLU parameter

        Function Returns
        -------
        tanh_output, logits_layer:
    '''
    with tf.variable_scope('generator', reuse=reuse_vars):
        # Defining the generator hidden layer
        hidden_layer_1 = tf.layers.dense(gen_z, num_hiddern_units, activation=None)

        # Feeding the output of hidden_layer_1 to leaky relu
        hidden_layer_1 = tf.maximum(hidden_layer_1, leaky_relu_alpha * hidden_layer_1)

        # Getting the logits and tanh layer output
        logits_layer = tf.layers.dense(hidden_layer_1, gen_out_dim, activation=None)
        tanh_output = tf.nn.tanh(logits_layer)

        return tanh_output, logits_layer


def discriminator(disc_input, num_hiddern_units=128, reuse_vars=False, leaky_relu_alpha=0.01):
    ''' Building the discriminator part of the network

        Function Arguments
        ---------
        disc_input : discrminator input tensor
        num_hiddern_units : Number of neurons/units in the hidden layer
        reuse_vars : Reuse variables with tf.variable_scope
        leaky_relu_alpha : leaky ReLU parameter

        Function Returns
        -------
        sigmoid_out, logits_layer:
    '''
    with tf.variable_scope('discriminator', reuse=reuse_vars):
        # Defining the generator hidden layer
        hidden_layer_1 = tf.layers.dense(disc_input, num_hiddern_units, activation=None)

        # Feeding the output of hidden_layer_1 to leaky relu
        hidden_layer_1 = tf.maximum(hidden_layer_1, leaky_relu_alpha * hidden_layer_1)

        logits_layer = tf.layers.dense(hidden_layer_1, 1, activation=None)
        sigmoid_out = tf.nn.sigmoid(logits_layer)

        return sigmoid_out, logits_layer



# size of discriminator input image
#28 by 28 will flattened to be 784
input_img_size = 784

# size of the generator latent vector
gen_z_size = 100

# number of hidden units for the generator and discriminator hidden layers
gen_hidden_size = 128
disc_hidden_size = 128

#leaky ReLU alpha parameter which controls the leak of the function
leaky_relu_alpha = 0.01

# smoothness of the label
label_smooth = 0.1

tf.reset_default_graph()

# creating the input placeholders for the discrminator and generator
real_discrminator_input, generator_input_z = inputs_placeholders(input_img_size, gen_z_size)

#Create the generator network
gen_model, gen_logits = generator(generator_input_z, input_img_size, gen_hidden_size, reuse_vars=False,  leaky_relu_alpha=leaky_relu_alpha)

# gen_model is the output of the generator
#Create the generator network
disc_model_real, disc_logits_real = discriminator(real_discrminator_input, disc_hidden_size, reuse_vars=False, leaky_relu_alpha=leaky_relu_alpha)
disc_model_fake, disc_logits_fake = discriminator(gen_model, disc_hidden_size, reuse_vars=True, leaky_relu_alpha=leaky_relu_alpha)


# calculating the losses of the discrimnator and generator
disc_labels_real = tf.ones_like(disc_logits_real) * (1 - label_smooth)
disc_labels_fake = tf.zeros_like(disc_logits_fake)

disc_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=disc_labels_real, logits=disc_logits_real)
disc_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=disc_labels_fake, logits=disc_logits_fake)

#averaging the disc loss
disc_loss = tf.reduce_mean(disc_loss_real + disc_loss_fake)

#averaging the gen loss
gen_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(disc_logits_fake),
        logits=disc_logits_fake))

# building the model optimizer

learning_rate = 0.002

# Getting the trainable_variables of the computational graph, split into Generator and Discrimnator parts
trainable_vars = tf.trainable_variables()
gen_vars = [var for var in trainable_vars if var.name.startswith("generator")]
disc_vars = [var for var in trainable_vars if var.name.startswith("discriminator")]

disc_train_optimizer = tf.train.AdamOptimizer().minimize(disc_loss, var_list=disc_vars)
gen_train_optimizer = tf.train.AdamOptimizer().minimize(gen_loss, var_list=gen_vars)

train_batch_size = 100
num_epochs = 100
generated_samples = []
model_losses = []

saver = tf.train.Saver(var_list=gen_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for e in range(num_epochs):
        for ii in range(mnist_dataset.train.num_examples // train_batch_size):
            input_batch = mnist_dataset.train.next_batch(train_batch_size)

            # Get images, reshape and rescale to pass to D
            input_batch_images = input_batch[0].reshape((train_batch_size, 784))
            input_batch_images = input_batch_images * 2 - 1

            # Sample random noise for G
            gen_batch_z = np.random.uniform(-1, 1, size=(train_batch_size, gen_z_size))

            # Run optimizers
            _ = sess.run(disc_train_optimizer,
                         feed_dict={real_discrminator_input: input_batch_images, generator_input_z: gen_batch_z})
            _ = sess.run(gen_train_optimizer, feed_dict={generator_input_z: gen_batch_z})

        # At the end of each epoch, get the losses and print them out
        train_loss_disc = sess.run(disc_loss,
                                   {generator_input_z: gen_batch_z, real_discrminator_input: input_batch_images})
        train_loss_gen = gen_loss.eval({generator_input_z: gen_batch_z})

        print("Epoch {}/{}...".format(e + 1, num_epochs),
              "Disc Loss: {:.3f}...".format(train_loss_disc),
              "Gen Loss: {:.3f}".format(train_loss_gen))

        # Save losses to view after training
        model_losses.append((train_loss_disc, train_loss_gen))

        # Sample from generator as we're training for viegenerator_inputs_zwing afterwards
        gen_sample_z = np.random.uniform(-1, 1, size=(16, gen_z_size))
        generator_samples = sess.run(
            generator(generator_input_z, input_img_size, reuse_vars=True),
            feed_dict={generator_input_z: gen_sample_z})

        generated_samples.append(generator_samples)
        saver.save(sess, './checkpoints/generator_ck.ckpt')

# Save training generator samples
with open('train_generator_samples.pkl', 'wb') as f:
    pkl.dump(generated_samples, f)


fig, ax = plt.subplots()
model_losses = np.array(model_losses)
plt.plot(model_losses.T[0], label='Disc loss')
plt.plot(model_losses.T[1], label='Gen loss')
plt.title("Model Losses")
plt.legend()


def view_generated_samples(epoch_num, g_samples):
    fig, axes = plt.subplots(figsize=(7, 7), nrows=4, ncols=4, sharey=True, sharex=True)

    print(gen_samples[epoch_num][1].shape)

    for ax, gen_image in zip(axes.flatten(), g_samples[0][epoch_num]):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        img = ax.imshow(gen_image.reshape((28, 28)), cmap='Greys_r')

    return fig, axes

# Load samples from generator taken while training
with open('train_generator_samples.pkl', 'rb') as f:
    gen_samples = pkl.load(f)

print('Testing the trained model by visualizing sample generated images...')
_ = view_generated_samples(-1, gen_samples)