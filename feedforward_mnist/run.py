"""
Use this file to run the MNIST network

Optional Flags:
--learning_rate
--max_steps
--hidden1
--hidden2
--batch_size
--train_dir
--fake_data
"""

from __future__ import print_function
import os.path
import time
import numpy
import tensorflow as tf
import input_data
import mnist

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer(
    'batch_size', 
    100, 
    'Batch size. Must divide evenly into the dataset sizes.'
)
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean(
    'fake_data', 
    False, 
    'If true, uses fake data for unit testing.'
)

def placeholder_inputs(batch_size):
    """
    Generate placeholder variables to represent the the input tensors. These placeholders 
    are used as inputs by the rest of the model building code and will be fed from the 
    downloaded data in the .run() loop, below.
    """

    images_placeholder = tf.placeholder(
        tf.float32, 
        shape = (batch_size, mnist.IMAGE_PIXELS)
    )

    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

    return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_placeholder, labels_placeholder):
    """
    Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
    """

    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)

    feed_dict = {
        images_placeholder : images_feed,
        labels_placeholder : labels_feed
    }

    return feed_dict

def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
    """
    one evaluation against the full epoch of data.
    """
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = int(data_set.num_examples / FLAGS.batch_size)
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(
            data_set,
            images_placeholder,
            labels_placeholder
        )
        true_count += sess.run(eval_correct, feed_dict = feed_dict)
    precision = float(true_count) / float(num_examples)
    print (
        '    Num examples: %d   Num correct: %d   Precision @ 1: %0.04f' %
        (num_examples, true_count, precision)
    )

def run_training():
    """
    Train MNIST for a number of steps
    """

    data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)

    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
        logits = mnist.inference(
            images_placeholder,
            FLAGS.hidden1,
            FLAGS.hidden2
        )
        loss = mnist.loss(logits, labels_placeholder)
        train_op = mnist.training(loss, FLAGS.learning_rate)
        eval_correct = mnist.evaluation(logits, labels_placeholder)
        
        summary_op = tf.merge_all_summaries()

        saver = tf.train.Saver()

        sess = tf.Session()

        init = tf.initialize_all_variables()
        sess.run(init)

        summary_writer = tf.training.summary_io.SummaryWriter(
            FLAGS.train_dir,
            graph_def = sess.graph_def
        )

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            feed_dict = fill_feed_dict(
                data_sets.train,
                images_placeholder,
                labels_placeholder
            )

            _, loss_value = sess.run([train_op, loss], feed_dict = feed_dict)

            duration = time.time() - start_time

            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                saver.save(sess, FLAGS.train_dir, global_step = step)

                print('Evaluating on training data...')
                do_eval(
                    sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.train
                )

                print('Evaluating on validation data...')
                do_eval(
                    sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.validation
                )

                print('Evaluating on test data...')
                do_eval(
                    sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.test
                )

def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()




