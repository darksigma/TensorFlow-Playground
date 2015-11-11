import tensorflow as tf

def linear(tensors, output_size, bias = True, bias_start = 0.0, scope = None):
    """
    Performs a linear map: sum_i(tensors[i] * W[i]), where W[i] is a variable.
    """

    if tensors is None:
        raise ValueError("Linear is expecting at least one tensor")
    
    with tf.variable_scope(scope or "Linear"):
        if not isinstance(tensors, (list, tuple)):
            shape = tensors.get_shape().as_list()
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
            tensor_size = shape[1]
            matrix = tf.get_variable("Matrix", [tensor_size, output_size])
            result = tf.matmul(tensors, matrix)
        else:
            total_tensor_size = 0
            shapes = [t.get_shape().as_list() for t in tensors]
            for shape in shapes:
                if len(shape) != 2:
                    raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
                total_tensor_size += shape[1]
            matrix = tf.get_variable("Matrix", [total_tensor_size, output_size])
            result = tf.matmul(tf.concat(1, tensors), matrix)

        if not bias:
            return result

        bias_terms = tf.get_variable(
            "Bias", 
            [output_size],
            initializer = tf.constant_initializer(bias_start)
        )

        return result + bias_terms


