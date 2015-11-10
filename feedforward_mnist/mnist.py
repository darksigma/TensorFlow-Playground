"""
This file describes the data flow model for a feedforward network 
with two hidden layers for the MNIST dataset. This is based on the
basic TensorFlow tutorial. 
"""

import math
import tensorflow as tf 

# MNIST has 10 possible digit classes
NUM_CLASSES = 10

# MNIST images are 28x28 pixels
IMAGE_PIXELS = 28 * 28

def inference(images, hidden1_units, hidden2_units):
    """
    Build the inference model for the MNIST network
    """

    # Hidden 1
    with tf.name_scope('hidden1') as scope:
        weights = tf.Variable(
            tf.truncated_normal(
                [IMAGE_PIXELS, hidden1_units],
                stddev = 1.0 / math.sqrt(float(IMAGE_PIXELS))
            ),
            name = 'weights'
        )

        biases = tf.Variable(
            tf.zeros([hidden1_units]),
            name = 'biases'
        )

        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    # Hidden 2
    with tf.name_scope('hidden1') as scope:
        weights = tf.Variable(
            tf.truncated_normal(
                [hidden1_units, hidden2_units],
                stddev = 1.0 / math.sqrt(float(hidden1_units))
            ),
            name = 'weights'
        )

        biases = tf.Variable(
            tf.zeros([hidden2_units]),
            name = 'biases'
        )

        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    # Softmax   
    with tf.name_scope('softmax_linear') as scope:
        weights = tf.Variable(
            tf.truncated_normal(
                [hidden2_units, NUM_CLASSES],
                stddev = 1.0/math.sqrt(float(hidden2_units))
            ),
            name = 'weights'
        )

        biases = tf.Variable(
            tf.zeros([NUM_CLASSES]),
            name = 'biases'
        )

        logits = tf.matmul(hidden2, weights) + biases

    return logits

def loss(logits, labels):
    """
    Building the loss model for the MNIST network
    """

    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concatenated = tf.concat(1, [indices, labels])
    one_hot_labels = tf.sparse_to_dense(
        concatenated,
        tf.pack([batch_size, NUM_CLASSES]),
        1.0,
        0.0
    )
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits,
        one_hot_labels,
        name = 'xentropy'
    )
    loss = tf.reduce_mean(cross_entropy, name = 'xentropy_mean')
    return loss

def training(loss, learning_rate):
    """
    Set up the training model for the MNIST network
    """

    # Add a summary for the loss
    tf.scalar_summary(loss.op.name, loss)

    # Create a gradient descent optimizer with the correct learning rate
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # Set up a variable to track the global step
    global_step = tf.Variable(0, name = 'global_step', trainable = False)

    # Apply gradient
    train_op = optimizer.minimize(loss, global_step = global_step)

    return train_op

def evaluation(logits, labels):
    """
    Set up a evaluation model for the MNIST network
    """

    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
