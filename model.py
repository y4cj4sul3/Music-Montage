import numpy as np
import tensorflow as tf

# TODO: import modules
#   - env
#   - enocder
#   - SB (soundbanj)
#   - decoder
import env
import encoder
import SB

# TODO: define inputs
x = ... # input
feature_x = f(x) # feature of the input

# algorithm
obs = env.reset()
z = encoder(obs) # z, symbolic form

y = decoder(z, SB)
loss = g(y) - g(obs)
reward = -loss
