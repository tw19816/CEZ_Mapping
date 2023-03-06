import tensorflow as tf
from aspp import ASPP

def deeplabv3plus(
    input_shape: tuple, batch_size: int, channels_low: int = 48
) -> tf.keras.Model:
    """Instance of the DeepLabV3+ encoder-decoder architecture with a modified 
    Xception backbone.
    
    Reference:
    - [Xception: Deep Learning with Depthwise Separable Convolutions](
        https://arxiv.org/abs/1610.02357) (CVPR 2017)
    
    Args:

    Returns:
        model (keras.Model) : A xception model instance.
    """
    # Entry flow
    img_input = tf.keras.Input(shape=input_shape, batch_size=batch_size)
    x = tf.keras.layers.Conv2D(
        32,
        (3, 3),
        strides=(2, 2),
        use_bias=False,
        padding="same",
        name="block1_conv1" 
    )(img_input)
    x = tf.keras.layers.BatchNormalization(name="block1_conv1_bn")(x)
    x = tf.keras.layers.Activation("relu", name="block1_conv1_act")(x)
    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding="same", use_bias=False, name="block1_conv2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="block1_conv2_bn")(x)
    x = tf.keras.layers.Activation("relu", name="block1_conv2_act")(x)

    residual = tf.keras.layers.Conv2D(
        128, (1, 1), strides=(2, 2), padding="same", use_bias=False, name="block2_skip"
    )(x)
    residual = tf.keras.layers.BatchNormalization()(residual)

    x = tf.keras.layers.SeparableConv2D(
        128, (3, 3), padding="same", use_bias=False, name="block2_sepconv1"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="block2_sepconv1_bn")(x)
    x = tf.keras.layers.Activation("relu", name="block2_sepconv2_act")(x)
    x = tf.keras.layers.SeparableConv2D(
        128, (3, 3), padding="same", use_bias=False, name="block2_sepconv2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="block2_sepconv2_bn")(x)
    x = tf.keras.layers.Activation("relu", name="block2_sepconv3stride_act")(x)    
    x = tf.keras.layers.SeparableConv2D(
        128, (3, 3), strides=(2, 2), padding="same", name="block2_sepconv3stride"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="block2_sepconv3stride_bn")(x)

    x = tf.keras.layers.add([x, residual])

    residual = tf.keras.layers.Conv2D(
        256, (1, 1), strides=(2, 2), padding="same", use_bias=False, name="block3_skip"
    )(x)
    residual = tf.keras.layers.BatchNormalization()(residual)

    x = tf.keras.layers.Activation("relu", name="block3_sepconv1_act")(x)
    x = tf.keras.layers.SeparableConv2D(
        256, (3, 3), padding="same", use_bias=False, name="block3_sepconv1"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="block3_sepconv1_bn")(x)
    x = tf.keras.layers.Activation("relu", name="block3_sepconv2_act")(x)
    x = tf.keras.layers.SeparableConv2D(
        256, (3, 3), padding="same", use_bias=False, name="block3_sepconv2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="block3_sepconv2_bn")(x)
    out_low = tf.keras.layers.Activation("relu", name="block3_sepconv3stride_act")(x)
    x = tf.keras.layers.SeparableConv2D(
        256, (3, 3), strides=(2, 2), padding="same", name="block3_sepconv3_stride"
    )(out_low)
    x = tf.keras.layers.BatchNormalization(name="block3_sepconv3stride_bn")(x)

    x = tf.keras.layers.add([x, residual])

    residual = tf.keras.layers.Conv2D(
        728, (1, 1), strides=(2, 2), padding="same", use_bias=False, name="block4_skip"
    )(x)
    residual = tf.keras.layers.BatchNormalization()(residual)

    x = tf.keras.layers.Activation("relu", name="block4_sepconv1_act")(x)
    x = tf.keras.layers.SeparableConv2D(
        728, (3, 3), padding="same", use_bias=False, name="block4_sepconv1"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="block4_sepconv1_bn")(
        x
    )
    x = tf.keras.layers.Activation("relu", name="block4_sepconv2_act")(x)
    x = tf.keras.layers.SeparableConv2D(
        728, (3, 3), padding="same", use_bias=False, name="block4_sepconv2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="block4_sepconv2_bn")(x)
    x = tf.keras.layers.Activation("relu", name="block4_sepconv3stride_act")(x)
    x = tf.keras.layers.SeparableConv2D(
        728, (3, 3), strides=(2, 2), padding="same", name="block4_sepconv3stride"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="block4_sepconv3stride_bn")(x)

    x = tf.keras.layers.add([x, residual])

    # Middle flow
    for i in range(16):
        residual = x
        prefix = "block" + str(i + 5)

        x = tf.keras.layers.Activation("relu", name=prefix + "_sepconv1_act")(x)
        x = tf.keras.layers.SeparableConv2D(
            728,
            (3, 3),
            padding="same",
            use_bias=False,
            name=prefix + "_sepconv1",
        )(x)
        x = tf.keras.layers.BatchNormalization(
            name=prefix + "_sepconv1_bn"
        )(x)
        x = tf.keras.layers.Activation("relu", name=prefix + "_sepconv2_act")(x)
        x = tf.keras.layers.SeparableConv2D(
            728,
            (3, 3),
            padding="same",
            use_bias=False,
            name=prefix + "_sepconv2",
        )(x)
        x = tf.keras.layers.BatchNormalization(
            name=prefix + "_sepconv2_bn"
        )(x)
        x = tf.keras.layers.Activation("relu", name=prefix + "_sepconv3_act")(x)
        x = tf.keras.layers.SeparableConv2D(
            728,
            (3, 3),
            padding="same",
            use_bias=False,
            name=prefix + "_sepconv3",
        )(x)
        x = tf.keras.layers.BatchNormalization(
            name=prefix + "_sepconv3_bn"
        )(x)

        x = tf.keras.layers.add([x, residual])

    # Exit flow
    residual = tf.keras.layers.Conv2D(
        1024, (1, 1), strides=(2, 2), padding="same", use_bias=False
    )(x)
    residual = tf.keras.layers.BatchNormalization()(residual)

    x = tf.keras.layers.Activation("relu", name="block13_sepconv1_act")(x)
    x = tf.keras.layers.SeparableConv2D(
        728, (3, 3), padding="same", use_bias=False, name="block13_sepconv1"
    )(x)
    x = tf.keras.layers.BatchNormalization(
        name="block13_sepconv1_bn"
    )(x)
    x = tf.keras.layers.Activation("relu", name="block13_sepconv2_act")(x)
    x = tf.keras.layers.SeparableConv2D(
        1024, (3, 3), padding="same", use_bias=False, name="block13_sepconv2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="block13_sepconv2_bn")(x)
    x = tf.keras.layers.Activation("relu", name="block13_sepconv3stride_act")(x)
    x = tf.keras.layers.SeparableConv2D(
        1024,
        (3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=False,
        name="block13_sepconv3stride"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="block13_sepconv3stride_bn")(x)

    x = tf.keras.layers.add([x, residual])

    x = tf.keras.layers.SeparableConv2D(
        1536, (3, 3), padding="same", use_bias=False, name="block14_sepconv1"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="block14_sepconv1_bn")(x)
    x = tf.keras.layers.Activation("relu", name="block14_sepconv1_act")(x)
    x = tf.keras.layers.SeparableConv2D(
        1536, (3, 3), padding="same", use_bias=False, name="block14_sepconv2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="block14_sepconv2_bn")(x)
    x = tf.keras.layers.Activation("relu", name="block14_sepconv2_act")(x)

    x = tf.keras.layers.SeparableConv2D(
        2048, (3, 3), padding="same", use_bias=False, name="block14_sepconv3"
    )(x)
    x = tf.keras.layers.BatchNormalization(
        name="block14_sepconv3_bn"
    )(x)
    out_backbone = tf.keras.layers.Activation("relu", name="block14_sepconv3_act")(x)

    # Decoder
    out_low = tf.keras.layers.Conv2D(
        channels_low,
        (1, 1),
        strides=(1, 1),
        padding="same",
        activation="relu",
        use_bias=False,
        name="dec_convl_low"
    )(out_low)

    # Intermediate output dimensions 
    tmp_target_h = tf.shape(out_low)[1]
    tmp_target_w = tf.shape(out_low)[2]

    out_decode = ASPP(256, (6, 12, 18), name="dec_aspp")(out_backbone)
    out_decode = tf.keras.layers.Resizing(
        tmp_target_h,
        tmp_target_w,
        interpolation="biliear",
        crop_to_aspect_ratio=False
    )(out_decode)

    out_decode = tf.keras.layers.concatenate([out_decode, out_low])

    out_decode = tf.keras.layers.Conv2D(
        256, (3, 3), padding="same", use_bias=False, name="dec1_conv1"
    )(out_decode)
    out_decode = tf.keras.layers.BatchNormalization(name="dec1_conv1_bn")(out_decode)
    out_decode = tf.keras.layers.Activation("relu", name="dec1_conv1_act")(out_decode)
    out_decode = tf.keras.layers.Conv2D(
        256, (3, 3), padding="same", use_bias=False, name="dec1_conv2"
    )(out_decode)
    out_decode = tf.keras.layers.BatchNormalization(name="dec1_conv2_bn")(out_decode)

    out_decode = tf.keras.layers.Activation("relu", name="dec1_conv2_act")(out_decode)

    # Create modified xception backbone
    backbone = tf.keras.Model(img_input, out_backbone)
    # Create full model
    model = tf.keras.Model(img_input, out_decode)
    return model




model = tf.keras.applications.Xception(include_top=False)
# class Conv(tf.keras.tf.keras.layers.Layer):
#     """A Xception convolutional layer with batch normalisation and ReLU activation
#     """
# create modified xception for backbone
# create decoder layer with two inputs (low and high level feature maps)
# create combined model from modified xception and decoder layer \
exit()