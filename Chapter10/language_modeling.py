#importing requried libraries
import numpy as np
import tensorflow as tf

from collections import namedtuple

#reading the Anna Karenina novel text file
with open('Anna_Karenina.txt', 'r') as f:
    textlines=f.read()

#Building the vocan and encoding the characters as integers
language_vocab = set(textlines)
vocab_to_integer = {char: j for j, char in enumerate(language_vocab)}
integer_to_vocab = dict(enumerate(language_vocab))
encoded_vocab = np.array([vocab_to_integer[char] for char in textlines], dtype=np.int32)


def generate_character_batches(data, num_seq, num_steps):
    '''Create a function that returns batches of size
       num_seq x num_steps from data.
    '''
    # Get the number of characters per batch and number of batches
    num_char_per_batch = num_seq * num_steps
    num_batches = len(data) // num_char_per_batch

    # Keep only enough characters to make full batches
    data = data[:num_batches * num_char_per_batch]

    # Reshape the array into n_seqs rows
    data = data.reshape((num_seq, -1))

    for i in range(0, data.shape[1], num_steps):
        # The input variables
        input_x = data[:, i:i + num_steps]

        # The output variables which are shifted by one
        output_y = np.zeros_like(input_x)

        output_y[:, :-1], output_y[:, -1] = input_x[:, 1:], input_x[:, 0]
        yield input_x, output_y

generated_batches = generate_character_batches(encoded_vocab, 15, 50)
input_x, output_y = next(generated_batches)

print('input\n', input_x[:10, :10])
print('\ntarget\n', output_y[:10, :10])


def build_model_inputs(batch_size, num_steps):
    # Declare placeholders for the input and output variables
    inputs_x = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
    targets_y = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')

    # define the keep_probability for the dropout layer
    keep_probability = tf.placeholder(tf.float32, name='keep_prob')

    return inputs_x, targets_y, keep_probability


def build_lstm_cell(size, num_layers, batch_size, keep_probability):
    ### Building the LSTM Cell using the tensorflow function
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(size)

    # Adding dropout to the layer to prevent overfitting
    drop_layer = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_probability)

    # Add muliple cells together and stack them up to oprovide a level of more understanding
    stakced_cell = tf.contrib.rnn.MultiRNNCell([drop_layer] * num_layers)
    initial_cell_state = lstm_cell.zero_state(batch_size, tf.float32)

    return lstm_cell, initial_cell_state


def build_model_output(output, input_size, output_size):
    # Reshaping output of the model to become a bunch of rows, where each row correspond for each step in the seq
    sequence_output = tf.concat(output, axis=1)
    reshaped_output = tf.reshape(sequence_output, [-1, input_size])

    # Connect the RNN outputs to a softmax layer
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal((input_size, output_size), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(output_size))

    # the output is a set of rows of LSTM cell outputs, so the logits will be a set
    # of rows of logit outputs, one for each step and sequence
    logits = tf.matmul(reshaped_output, softmax_w) + softmax_b

    # Use softmax to get the probabilities for predicted characters
    model_out = tf.nn.softmax(logits, name='predictions')

    return model_out, logits


def model_loss(logits, targets, lstm_size, num_classes):
    # convert the targets to one-hot encoded and reshape them to match the logits, one row per batch_size per step
    output_y_one_hot = tf.one_hot(targets, num_classes)
    output_y_reshaped = tf.reshape(output_y_one_hot, logits.get_shape())

    # Use the cross entropy loss
    model_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=output_y_reshaped)
    model_loss = tf.reduce_mean(model_loss)
    return model_loss


def build_model_optimizer(model_loss, learning_rate, grad_clip):
    # define optimizer for training, using gradient clipping to avoid the exploding of the gradients
    trainable_variables = tf.trainable_variables()
    gradients, _ = tf.clip_by_global_norm(tf.gradients(model_loss, trainable_variables), grad_clip)

    # Use Adam Optimizer
    train_operation = tf.train.AdamOptimizer(learning_rate)
    model_optimizer = train_operation.apply_gradients(zip(gradients, trainable_variables))

    return model_optimizer


class CharLSTM:
    def __init__(self, num_classes, batch_size=64, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False):

        # When we're using this network for generating text by sampling, we'll be providing the network with
        # one character at a time, so providing an option for it.
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()

        # Build the model inputs placeholders of the input and target variables
        self.inputs, self.targets, self.keep_prob = build_model_inputs(batch_size, num_steps)

        # Building the LSTM cell
        lstm_cell, self.initial_state = build_lstm_cell(lstm_size, num_layers, batch_size, self.keep_prob)

        ### Run the data through the LSTM layers
        # one_hot encode the input
        input_x_one_hot = tf.one_hot(self.inputs, num_classes)

        # Runing each sequence step through the LSTM architecture and finally collectting the outputs
        outputs, state = tf.nn.dynamic_rnn(lstm_cell, input_x_one_hot, initial_state=self.initial_state)
        self.final_state = state

        # Get softmax predictions and logits
        self.prediction, self.logits = build_model_output(outputs, lstm_size, num_classes)

        # Loss and optimizer (with gradient clipping)
        self.loss = model_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_model_optimizer(self.loss, learning_rate, grad_clip)

batch_size = 100        # Sequences per batch
num_steps = 100         # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.001   # Learning rate
keep_probability = 0.5         # Dropout keep probability


print('Starting the training process...')
epochs = 5

# Save a checkpoint N iterations
save_every_n = 100

LSTM_model = CharLSTM(len(language_vocab), batch_size=batch_size, num_steps=num_steps,
                      lstm_size=lstm_size, num_layers=num_layers,
                      learning_rate=learning_rate)

saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Use the line below to load a checkpoint and resume training
    # saver.restore(sess, 'checkpoints/______.ckpt')
    counter = 0
    for e in range(epochs):
        # Train network
        new_state = sess.run(LSTM_model.initial_state)
        loss = 0
        for x, y in generate_character_batches(encoded_vocab, batch_size, num_steps):
            counter += 1
            start = time.time()
            feed = {LSTM_model.inputs: x,
                    LSTM_model.targets: y,
                    LSTM_model.keep_prob: keep_probability,
                    LSTM_model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([LSTM_model.loss,
                                                 LSTM_model.final_state,
                                                 LSTM_model.optimizer],
                                                feed_dict=feed)

            end = time.time()
            print('Epoch number: {}/{}... '.format(e + 1, epochs),
                  'Step: {}... '.format(counter),
                  'loss: {:.4f}... '.format(batch_loss),
                  '{:.3f} sec/batch'.format((end - start)))

            if (counter % save_every_n == 0):
                saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))

    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))

print('Training is done...')


#Defining helper functions for sampling from the network
def pick_top_n_characters(preds, vocab_size, top_n_chars=4):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n_chars]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def sample_from_LSTM_output(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    samples = [c for c in prime]
    LSTM_model = CharLSTM(len(language_vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(LSTM_model.initial_state)
        for char in prime:
            x = np.zeros((1, 1))
            x[0, 0] = vocab_to_integer[char]
            feed = {LSTM_model.inputs: x,
                    LSTM_model.keep_prob: 1.,
                    LSTM_model.initial_state: new_state}
            preds, new_state = sess.run([LSTM_model.prediction, LSTM_model.final_state],
                                        feed_dict=feed)

        c = pick_top_n_characters(preds, len(language_vocab))
        samples.append(integer_to_vocab[c])

        for i in range(n_samples):
            x[0, 0] = c
            feed = {LSTM_model.inputs: x,
                    LSTM_model.keep_prob: 1.,
                    LSTM_model.initial_state: new_state}
            preds, new_state = sess.run([LSTM_model.prediction, LSTM_model.final_state],
                                        feed_dict=feed)

            c = pick_top_n_characters(preds, len(language_vocab))
            samples.append(integer_to_vocab[c])

    return ''.join(samples)

print('Loading latest checkpoint..')
checkpoint = tf.train.latest_checkpoint('checkpoints')

print('Sampling text frm the trained model....')
sampled_text = sample_from_LSTM_output(checkpoint, 2000, lstm_size, len(language_vocab), prime="Far")
print(sampled_text)