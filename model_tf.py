import tensorflow as tf
import cvnn.layers as complex_layers


def make_model():
    model = tf.keras.models.Sequential()
    model.add(complex_layers.ComplexInput(input_shape=(100, 100)))
    model.add(complex_layers.ComplexFlatten())
    model.add(complex_layers.ComplexDense(20, activation='cart_tanh',
              kernel_regularizer=tf.keras.regularizers.L2(0.1)))
    model.add(complex_layers.ComplexDense(1, activation='convert_to_real_with_abs'))
    return model
