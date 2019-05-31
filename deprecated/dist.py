import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from tf import placeholder
from tf.contrib.layers import fully_connected
from tf.nn import embedding_lookup

from tf.losses import mean_squared_error
from tf.train import AdamOptimizer

from tfp.distributions import RelaxedOneHotCategorical

batch_size = 32

n_seg = 100 # num. of segments
m_cate = 50 # num. of category of clips
seg_len = 3 # segment length

steps = 1000
temperature = 0.5 # gumble softmax
sess = tf.InteractiveSession()

# load data 
X = np.random.random_sample(batch_size, n, seg_len) * 5 + 50 # n segments
SB = np.random.random_sample(batch_size, m_cate, seg_len) * 5 + 50 # m categories of clips

# model
# ----------------------------------------
x_input = placeholder(tf.float32, shape=(None, seg_len, n_seg))
sb_input = placeholder(tf.float32, shape(None))

# generating model for the dist
z = tf.contrib.layers.fully_connected(x_input, m_cate)
z_onehot = RelaxedOneHotCategorical(temperature, probs=z)

# embedding layer
y = embedding_lookup(sb_input, z_onehot)

# loss and optimizer
loss = mean_squared_error(x_input, y)
opt = AdamOptimizer(learning_rate=0.001)
train_op = opt.minimize(loss)

# main
# ----------------------------------------
tf.initialize_all_variables()

for i in range(100):
    y_output, _ = sess.run([y, train_op], feed_dict={
        x_input: X, sb_input: SB
    })

    print(X[0], y_output[0])
