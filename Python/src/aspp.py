import tensorflow as tf

class Atrous2D(tf.keras.layers.Layer):
    """An atrous convolution layer followed by a batch normalisation and ReLU.
    """
    def __init__(
        self,
        filters: int,
        dilation_rate: tuple,
        kernel: tuple = (3, 3),
        padding: str = "valid",
        name: str = "Atrous2D"
    ):
        """Creates a atrous convolution layer followed by a batch normalisation
            and a ReLU activation layer.

            Args:
                filters (int) : The number of filters/output-channels.
                dilation_rate (tuple (int)) : Two integers specifying x and y 
                    dihalation/atrous rate.
                kernel (tuple (int)) : Two intergers specifiying the dimensions
                    of the kernel.
                padding (str) : See keras.layers.Conv2D for API.
                name (str) : The name of the layer.
        """
        super(Atrous2D, self).__init__(name=name)
        self._atrous_conv = tf.keras.layers.Conv2D(
            filters,
            kernel,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=False
        )
        self._batch_norm = tf.keras.layers.BatchNormalization(name=name + "_bn")
        self._activation = tf.keras.layers.Activation("relu", name=name + "_act")
 
    def call(self, input_tensor: tf.Tensor, training: bool = False):
        """Performs a forward pass.

        Args:
            input_tensor (tensorflow.Tensor) : Input to perform forward pass on.
            training (bool) : A boolean flag indicating whether training 
                behavior should be used (default: False).
        
        Returns:
            out (tensorflow.Tensor) : The output tensor.
        """
        out = self._atrous_conv(input_tensor, training=training)
        out = self._batch_norm(out, training=training)
        out = self._activation(out, training=training)
        return out
        

class ASPP(tf.keras.layers.Layer):
    """An atrous spatial pyramid pooling layer."""
    def __init__(
        self,
        output_channels,
        atrous_rates,
        name: str = "ASPP"
    ):
        """Create an ASPP layer that performs spatial pyramid pooling and 
        with global image pooling and returns the concatenated results.
        
        Args:
            name (str) : A string specifiying the name of this layer

        Returns:

        """
        super(ASPP, self).__init__(name=name)
        
        if len(atrous_rates) != 3:
            raise ValueError(
                "The ASPP layer needs exactly 3 atrous rates but %d were given" %
                len(atrous_rates)
            )
        self._aspp_conv_1_1 = Atrous2D(
            output_channels, dilation_rate=1, kernel=(1, 1), name="aspp_conv_1_1"
        )
        rate1, rate2, rate3 = atrous_rates
        self._aspp_conv1 = Atrous2D(
            output_channels, dilation_rate=rate1, padding="same", name="aspp_conv1"
        )
        self._aspp_conv2 = Atrous2D(
            output_channels, dilation_rate=rate2, padding="same", name="aspp_conv2"
        )
        self._aspp_conv3 = Atrous2D(
            output_channels, dilation_rate=rate3, padding="same", name="aspp_conv3"
        )
        self._aspp_global = tf.keras.layers.GlobalAveragePooling2D(
            keepdims=True, name="aspp_glob"
        )
    
    def call(self, input_tensor: tf.Tensor, training: bool = False):
        """Performs a forward pass.

        Args:
            input_tensor (tensorflow.Tensor) : Input to perform forward pass on.
            training (bool) : A boolean flag indicating whether training 
                behavior should be used (default: False).
        
        Returns:
            out (tensorflow.Tensor) : The output tensor.
        """
        conv_1_1 = self._aspp_conv_1_1(input_tensor, training=training)
        conv1 = self._aspp_conv1(input_tensor, training=training)
        conv2 = self._aspp_conv2(input_tensor, training=training)
        conv3 = self._aspp_conv3(input_tensor, training=training)
        glob = self._aspp_global(input_tensor, training=training)
        
        # Resize global output
        target_h = input_tensor.get_shape().as_list()[1]
        target_w = input_tensor.get_shape().as_list()[2]
        glob = tf.keras.layers.Resizing(
            target_h,
            target_w,
            interpolation="bilinear",
            crop_to_aspect_ratio=False
        )(glob)
        out = tf.keras.layers.concatenate([conv_1_1, conv1, conv2, conv3, glob])
        return out