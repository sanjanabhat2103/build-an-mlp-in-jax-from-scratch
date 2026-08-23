"""
Build an MLP in JAX from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_prng_key
import jax
import jax.numpy as jnp


def make_prng_key(seed):
    return jax.random.PRNGKey(seed)

# Step 2 - split_prng_key
import jax

def split_prng_key(key, num):
    return jax.random.split(key, num)

# Step 3 - sample_normal_matrix
import jax
import jax.numpy as jnp

def sample_normal_matrix(key, shape):
    return jax.random.normal(key, shape)

# Step 4 - sample_input_features
import jax
import jax.numpy as jnp

def sample_input_features(key, batch_size, num_features):
    """Sample a (batch_size, num_features) standard-normal feature batch."""
    return sample_normal_matrix(key, (batch_size, num_features))

# Step 5 - assign_class_labels
def assign_class_labels(inputs, num_classes):
    return jnp.argmax(inputs[:, :num_classes], axis = 1).astype(jnp.int32)

# Step 6 - one_hot_encode_labels
def one_hot_encode_labels(labels, num_classes):
    return jax.nn.one_hot(labels, num_classes)

# Step 7 - init_linear_layer
import jax
import jax.numpy as jnp

def init_linear_layer(key, in_dim, out_dim, scale=0.1):
    """Return {'W': (in_dim, out_dim), 'b': (out_dim,)} for one dense layer."""
    W = scale * sample_normal_matrix(key, (in_dim, out_dim))
    b = jnp.zeros(out_dim)
    return {'W': W, 'b': b}

# Step 8 - init_mlp_params
def init_mlp_params(key, layer_sizes, scale=0.1):
    keys = split_prng_key(key, len(layer_sizes) - 1)
    layers = [init_linear_layer(layer_key, in_dim, out_dim, scale) for layer_key, (in_dim, out_dim) in zip(keys, zip(layer_sizes[: -1], layer_sizes[1: ]))]
    return layers

# Step 9 - linear_forward
def linear_forward(x, layer_params):
   W = layer_params["W"]
   b = layer_params["b"]
   return x @ W + b

# Step 10 - relu_activation
import jax.numpy as jnp


def relu_activation(x):
    """Apply the ReLU activation elementwise to a JAX array."""
    return jnp.maximum(0, x)

# Step 11 - softmax_probabilities
import jax.numpy as jnp

def softmax_probabilities(logits):
    shifted = logits - jnp.max(logits, axis = -1, keepdims = True)
    exp_logits = jnp.exp(shifted)
    return exp_logits / jnp.sum(exp_logits, axis = -1, keepdims = True)

# Step 12 - mlp_forward
def mlp_forward(params, x):
    h = x
    for layer in params[: -1]:
        h = linear_forward(h, layer)
        h = relu_activation(h)
    return linear_forward(h, params[-1])

# Step 13 - log_softmax_logits
def log_softmax_logits(logits):
    shifted = logits - jnp.max(logits, axis = -1, keepdims = True)
    return shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis = -1, keepdims = True))

# Step 14 - cross_entropy_loss (not yet solved)
# TODO: implement

# Step 15 - classification_accuracy (not yet solved)
# TODO: implement

# Step 16 - loss_fn_of_params (not yet solved)
# TODO: implement

# Step 17 - compute_param_grads (not yet solved)
# TODO: implement

# Step 18 - sgd_update_params (not yet solved)
# TODO: implement

# Step 19 - training_step (not yet solved)
# TODO: implement

# Step 20 - train_mlp (not yet solved)
# TODO: implement

# Step 21 - predict_classes (not yet solved)
# TODO: implement

