import math
import tensorflow as tf 
sys.path.insert(0, os.path.abspath(".."))
from tensorflow_playground.rnn_blocks.linear import linear

class RNNLayer(object):
    """
    Abstract class representing a layer of RNN 
    """

    def __call__(self, inputs, state, scope = None):
        """
        Run the RNN on the inputs given the starting state. Returns the output
        and the new state.
        """

        raise NotImplementedError("Abstract Method")

    @property
    def input_size(self):
        """
        The size of the input accepted by this layer
        """

        raise NotImplementedError("Abstract Method")

    @property
    def output_size(self):
        """
        The size of the output emitted by this layer
        """

        raise NotImplementedError("Abstract Method")

    @property
    def state_size(self):
        """
        The size of the state of this layer
        """

        raise NotImplementedError("Abstract Method")

    @property
    def zero_state(self, batch_size, dtype):
        """
        Return state tensor (shape [batch_size x state_size]) filled with 0.
        """

        zeros = tf.zeros(tf.pack([batch_size, self.state_size]))
        return tf.reshape(zeros, [batch_size, self.state_size])

class BasicRNNLayer(RNNLayer):
    """
    Simplest possible RNN layer implementation
    """

    def __init__(self, num_units):
        self._num_units = num_units

    @property 
    def input_size(self):
        return self._num_units

    @property 
    def output_size(self):
        return self._num_units

    @property 
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope = None):
        with tf.variable_scope(scope or type(self).__name__):
            output = tf.tanh(linear([inputs, state], self._num_units, bias = True))
        return output, output

class GRULayer(RNNLayer):
    """
    Gated Recurrent Layer (cf.http://arxiv.org/abs/1406.1078)
    """

    def __init__(self, num_units):
        self._num_units = num_units

    @property 
    def input_size(self):
        return self._num_units

    @property 
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope = None):
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("Gates"):
                reset, update = tf.split(
                    1,
                    2,
                    linear(
                        [inputs, states], 
                        2 * self._num_units,
                        bias = True,
                        bias_start = 1.0
                    )
                )
                reset, update = tf.sigmoid(reset), tf.sigmoid(update)

            with tf.variable_scope("Candidate"):
                candidate = linear(
                    [inputs, reset * state],
                    self._num_units,
                    bias = True
                )
                candidate = tf.tanh(candidate)

            new_state = update * state + (1 - update) * candidate

            return new_state, new_state

class BasicLSTMLayer(RNNLayer):
    """
    Basic LSTM Layer (cf. http://arxiv.org/pdf/1409.2329v5.pdf)
    """

    def __init__(self, num_units, forget_bias = 1.0):
        self._num_units = num_units
        self._forget_bias = forget_bias

    @property 
    def input_size(self):
        return self._num_units

    @property 
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return 2 * self._num_units

    def __call__(self, inputs, state, scope = None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = tf.split(1, 2, state)
            concat = linear(
                [inputs, h], 
                4 * self._num_units, 
                bias = True
            )

            i, j, f, o = tf.split(1, 4, concat)

            new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)
            new_state = tf.concat(1, [new_c, new_h])

        return new_h, new_state

class DropoutWrapper(RNNLayer):
    """
    Wrapper for applying dropout to the inputs and ouptuts for an RNN layer
    """

    def __init__(self, layer, input_keep_prob = 1.0, output_keep_prob = 1.0, seed = None):
        if not isinstance(layer, RNNLayer):
            raise TypeError("Parameter isn't an RNNLayer")

        if isinstance(input_keep_prob. float) and not (input_keep_prob >= 0.0 and input_keep_prob <= 1.0):
            raise ValueError("Parameter input_keep_prob must be between 0 and 1: %d" % input_keep_prob)

        if isinstance(output_keep_prob. float) and not (output_keep_prob >= 0.0 and output_keep_prob <= 1.0):
            raise ValueError("Parameter input_keep_prob must be between 0 and 1: %d" % output_keep_prob)

        self._layer = layer
        self._input_keep_prob = input_keep_prob
        self._output_keep_prob = output_keep_prob
        self._seed = seed

        @property 
        def input_size(self):
            return self._layer.input_size

        @property 
        def output_size(self):
            return self._layer.output_size

        @property
        def state_size(self):
            return self._layer.state_size

        def __call__(self, inputs, state):
            if not isinstance(self._input_keep_prob, float) or self._input_keep_prob < 1:
                inputs = tf.nn.dropout(inputs, self._input_keep_prob, seed=self._seed)
            output, new_state = self._layer(inputs, state)
            if not isinstance(self._output_keep_prob, float) 

class MultiRNNLayer(RNNLayer):
    """
    A stack of multiple RNN Layers
    """

    def __init__(self, layers):
        if not layers:
            raise ValueError("Must have at least one layer")

        for i in xrange(len(layers) - 1):
            if layers[i + 1].input_size != layers[i].output_size:
                raise ValueError("The input size of each layer must be match the output of the previous")

        self._layers = layers

    @property 
    def input_size(self):
        return self._layers[0].input_size

    @property 
    def output_size(self):
        return self._layers[-1].output_size

    @property 
    def state_size(self):
        return sum([layer.state_size for layer in self._layers])

    def __call__(self, inputs, scope = None):
        with tf.variable_scope(scope or type(self).__name__):
            cur_state_pos = 0
            cur_inp = inputs
            new_states = []
            for i, layer in enumerate(self._layers):
                with tf.variable_scope("Cell%d" % i):
                    cur_state = tf.slice(state, [0, cur_state_pos], [-1, layer.state_size])
                    cur_state_pos += layer.state_size
                    cur_inp, new_state = layer(cur_inp, cur_state)
                    new_states.append(new_state)
        return cur_inp, tf.concat(1, new_states)


