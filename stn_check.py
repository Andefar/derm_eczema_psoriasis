import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from spatial_transformer import transformer

height = 1200
width = 1200
I = Image.open("data_test/eczema_data/patient_12646_0.jpg")
I = I.resize((height, width))
I = np.asarray(I) / 255.
I = I.astype('float32')

num_batch = 10


# out dims
out_H = 600
out_W = 600
out_dims = (out_H, out_W)

# repeat Image 3 times to simulate batches.
input_img = np.concatenate([[I.copy() for _ in range(num_batch)]])

B, H, W, C = input_img.shape
print("Input Img Shape: {}".format(input_img.shape))
#theta = np.array([[np.cos(45), -np.sin(45), 0], [np.sin(45), np.cos(45), 0]])
#theta = np.array([[0.5,0.5,0], [0.5,0.5,0]])
theta = np.array([ 0.44999999,  0.      ,    0.     ,     0.   ,       0.44999999 ,  0.        ])

x = tf.placeholder(tf.float32, [None, H, W, C])

# create localisation network and convolutional layer
with tf.variable_scope('spatial_transformer_0'):

    # create a fully-connected layer with 6 output nodes
    n_fc = 6
    W_fc1 = tf.Variable(tf.zeros([H*W*C, n_fc]), name='W_fc1')

    # affine transformation
    theta = theta.astype('float32')
    theta = theta.flatten()

    b_fc1 = tf.Variable(initial_value=theta, name='b_fc1')
    h_fc1 = tf.matmul(tf.zeros([B, H*W*C]), W_fc1) + b_fc1
    h_trans = transformer(x, h_fc1, out_size=out_dims)

# run session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
y = sess.run(h_trans, feed_dict={x: input_img})
print(y.shape)
y = np.reshape(y, (B, out_H, out_W, C))

print(y[0])

plt.imshow(y[0])
plt.show()