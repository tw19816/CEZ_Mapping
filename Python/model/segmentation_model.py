import tensorflow as tf
from Python.model.aspp import ASPP

def deeplabv3plus(
    input_shape: tuple,
    batch_size: int,
    out_channels: int,
    channels_low: int = 48,
    channels_high: int = 2048,
    middle_repeat: int = 16
) -> tf.keras.Model:
    """Instance of the DeepLabV3+ encoder-decoder architecture with a modified 
    Xception backbone.
    
    Reference:
    - [Xception: Deep Learning with Depthwise Separable Convolutions](
        https://arxiv.org/abs/1610.02357) (CVPR 2017)
    
    Args:
        input_shape (tuple (int)) : Three integers used to specify the
            dimensions of the input tensor in terms of x, y, channels 
            respectively.
        batch_size (int) : Number of inputs per batch.
        out_channels (int) : Number of output channels.
        channels_low (int) : Number of channels to down-sample low level
            features to before combining with high-level features in decoder.
        channels_high (int) : Number of channels to down-sample high level 
            features to before combining with low-level features in decoder.
            This is the number of channels from the Xception backbone.
        middle_repeat (int) : Number of times to repeat middle Xception flow.

    Returns:
        model (keras.Model) : A DeepLabV3+ instance with an Xception backbone
            and 256 output channels.
    """
    # Entry flow
    img_input = tf.keras.Input(shape=input_shape, batch_size=batch_size)
    # img_input = tf.keras.Input(shape=input_shape)
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
    for i in range(middle_repeat):
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

    x = tf.keras.layers.Activation("relu", name="block21_sepconv1_act")(x)
    x = tf.keras.layers.SeparableConv2D(
        728, (3, 3), padding="same", use_bias=False, name="block21_sepconv1"
    )(x)
    x = tf.keras.layers.BatchNormalization(
        name="block21_sepconv1_bn"
    )(x)
    x = tf.keras.layers.Activation("relu", name="block21_sepconv2_act")(x)
    x = tf.keras.layers.SeparableConv2D(
        1024, (3, 3), padding="same", use_bias=False, name="block21_sepconv2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="block21_sepconv2_bn")(x)
    x = tf.keras.layers.Activation("relu", name="block21_sepconv3stride_act")(x)
    x = tf.keras.layers.SeparableConv2D(
        1024,
        (3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=False,
        name="block21_sepconv3stride"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="block21_sepconv3stride_bn")(x)

    x = tf.keras.layers.add([x, residual])

    x = tf.keras.layers.SeparableConv2D(
        1536, (3, 3), padding="same", use_bias=False, name="block22_sepconv1"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="block22_sepconv1_bn")(x)
    x = tf.keras.layers.Activation("relu", name="block22_sepconv1_act")(x)
    x = tf.keras.layers.SeparableConv2D(
        1536, (3, 3), padding="same", use_bias=False, name="block22_sepconv2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="block22_sepconv2_bn")(x)
    x = tf.keras.layers.Activation("relu", name="block22_sepconv2_act")(x)

    x = tf.keras.layers.SeparableConv2D(
        channels_high, (3, 3), padding="same", use_bias=False, name="block22_sepconv3"
    )(x)
    x = tf.keras.layers.BatchNormalization(
        name="block22_sepconv3_bn"
    )(x)
    out_backbone = tf.keras.layers.Activation("relu", name="block22_sepconv3_act")(x)

    # Decoder
    out_low = tf.keras.layers.Conv2D(
        channels_low,
        (1, 1),
        strides=(1, 1),
        activation="relu",
        use_bias=True,
        name="dec_convl_low"
    )(out_low)

    # Intermediate output dimensions 
    tmp_target_h = out_low.get_shape().as_list()[1]
    tmp_target_w = out_low.get_shape().as_list()[2]


    out_decode = ASPP(256, (6, 12, 18), name="dec_aspp")(out_backbone)
    out_decode = tf.keras.layers.Conv2D(
        256, (1, 1), use_bias=False, name="dec_conv1_high"
    )(out_decode)
    out_decode = tf.keras.layers.BatchNormalization(
        name="dec_conv1_high_bn"
    )(out_decode)
    out_decode = tf.keras.layers.Activation(
        "relu", name="dec_conv1_high_act"
    )(out_decode)

    out_decode = tf.keras.layers.Resizing(
        tmp_target_h,
        tmp_target_w,
        interpolation="bilinear",
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

    out_decode = tf.keras.layers.Conv2D(
        out_channels, (1, 1), use_bias=False, name="dec1_output"
    )(out_decode)
    out_decode = tf.keras.layers.Resizing(
        input_shape[1],
        input_shape[1],
        interpolation="bilinear",
        crop_to_aspect_ratio=False
    )(out_decode)

    # Create full model
    model = tf.keras.Model(img_input, out_decode)
    return model