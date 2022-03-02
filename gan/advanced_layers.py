import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras import layers


def residual_block(n_filters, input_layer):
    x = layers.Conv2D(
        filters=n_filters, kernel_size=(3, 3), padding='same'
    )(input_layer)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        filters=n_filters, kernel_size=(3, 3), padding='same'
    )(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Concatenate()([x, input_layer])

    return x


def subpixel_layer(x, r):
    _, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, (-1, h, w, r, r))
    x = tf.transpose(x, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    x = tf.split(axis=1, num_or_size_splits=h, value=x)  # a, [bsize, b, r, r]
    x = tf.concat(axis=2, values=[tf.squeeze(x, axis=1) for x in x])  # bsize, b, a*r, r
    x = tf.split(axis=1, num_or_size_splits=w, value=x)  # b, [bsize, a*r, r]
    x = tf.concat(axis=2, values=[tf.squeeze(x, axis=1) for x in x])  # bsize, a*r, b*r
    return tf.reshape(x, (-1, h * r, w * r, 1))


def subpixel_up_sampling(inputs, r):
    up_sampled_layer = tf.nn.depth_to_space(input=inputs, block_size=r)
    return up_sampled_layer


def densely_connected_residual_block(inputs):
    _, _, _, c = inputs.get_shape().as_list()
    growth_rate = int(c / 2)
    x1 = layers.Conv2D(
        filters=growth_rate, kernel_size=(3, 3),
        strides=(1, 1), padding='same'
    )(inputs)
    x1 = layers.PReLU()(x1)

    x2_inputs = layers.Concatenate()([x1, inputs])
    x2 = layers.Conv2D(
        filters=growth_rate, kernel_size=(3, 3),
        strides=(1, 1),padding='same'
    )(x2_inputs)
    x2 = layers.PReLU()(x2)

    x3_inputs = layers.Concatenate()([x1, x2, inputs])
    x3 = layers.Conv2D(
        filters=c, kernel_size=(3, 3),
        strides=(1, 1), padding='same'
    )(x3_inputs)
    x3 = layers.PReLU()(x3)

    return x3


def channel_attention_block(inputs: tf.Tensor, r: int):
    num_channels = inputs.shape.as_list()[-1]
    global_pooling = tf.reduce_mean(
        input_tensor=inputs, axis=[1, 2], keepdims=True,
    )
    x = layers.Conv2D(
        filters=int(num_channels / r),
        kernel_size=(1, 1),
    )(global_pooling)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        filters=num_channels,
        kernel_size=(1, 1),
    )(x)
    attention_weights = layers.Activation('sigmoid')(x)
    output = inputs * attention_weights
    return output
