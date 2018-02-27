import numpy as np
import tensorflow as tf

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 6)

input_values = np.arange(0.0, 5.0, 0.1)

print('Input Values...')
print(input_values)

print('defining the linear regression equation...')
weight=1
bias=0


output = weight*input_values + bias

print('Plotting the output...')
plt.plot(input_values,output)
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

print('Implementing the linear regression model in TensorFlow...')

input_values = np.random.rand(100).astype(np.float32)
output_values = input_values * 2 + 3
output_values = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.1))(output_values)

print('Sample from the input and output values of the linear regression model...')
print(list(zip(input_values,output_values))[5:10])

weight = tf.Variable(1.0)
bias = tf.Variable(0.2)

predicted_vals = weight * input_values + bias

print('Defining the model loss and optimizer...')
model_loss = tf.reduce_mean(tf.square(predicted_vals - output_values))
model_optimizer = tf.train.GradientDescentOptimizer(0.5)


train = model_optimizer.minimize(model_loss)


print('Initializing the global variables...')
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

train_data = []

for step in range(100):

    evals = sess.run([train,weight,bias])[1:]

    if step % 5 == 0:

       print(step, evals)

       train_data.append(evals)


print('Plotting the data points with their corresponding fitted line...')
converter = plt.colors
cr, cg, cb = (1.0, 1.0, 0.0)

for f in train_data:

    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)

    if cb > 1.0: cb = 1.0

    if cg < 0.0: cg = 0.0

    [a, b] = f
    f_y = np.vectorize(lambda x: a*x + b)(input_values)
    line = plt.plot(input_values, f_y)
    plt.setp(line, color=(cr,cg,cb))

plt.plot(input_values, output_values, 'ro')
green_line = mpatches.Patch(color='red', label='Data Points')
plt.legend(handles=[green_line])
plt.show()