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
