# Importing necessary layers from TensorFlow's Keras API
from tensorflow.keras import layers, models
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Flatten, TimeDistributed, Reshape, Conv2DTranspose, Lambda, Concatenate, Activation, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras import applications, layers, models
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# Initial attempts to implement CNNs utilizing Conv2D and MaxPooling layers for audio spectrogram processing
# did not yield the expected accuracy, probably because of the high amount of details that needs to be reconstructed
# in the case of audio linear spectograms. To overcome this, we explored using a pre-trained model that offers both
# computational efficiency and higher accuracy. This exploration led us to adopt MobileNetV2 as our encoder model.
# Instead of employing traditional Upsampling via ConvolutionTranspose for the decoder, we decided to adopt a more
# sophisticated approach. Inspired by the MobileNetV2 architecture, specifically its use of Inverted Residual Blocks
# (https://arxiv.org/abs/1801.04381), we aimed to reverse engineer these operations for our decoding process.
# Following insights from "MobileNetV2 Autoencoder: An Efficient Approach for Feature Extraction and Image Reconstruction"
# (https://medium.com/@abbesnessim/mobilenetv2-autoencoder-an-efficient-approach-for-feature-extraction-and-image-reconstruction-9c70ba58947a),
# we implemented a decoder that utilizes the InvertedResidualBlockTranspose class to mirror the encoding process effectively.
class InvertedResidualBlockTranspose(layers.Layer):
    def __init__(self, in_channels, out_channels, expansion, stride, name, **kwargs):
        """
        Initializes the InvertedResidualBlockTranspose layer.
        """
        super(InvertedResidualBlockTranspose, self).__init__(name=name, **kwargs)

        # Expansion layer uses 1x1 convolutions to expand the channel dimension
        # according to the expansion factor.
        self.expansion_layer = models.Sequential([
            layers.Conv2D(filters=in_channels * expansion, kernel_size=1, padding='same', use_bias=False, name=f"{name}_expansion_conv"),
            layers.BatchNormalization(name=f"{name}_expansion_bn"),
            layers.ReLU(max_value=6.0, name=f"{name}_expansion_relu")
        ], name=f"{name}_expansion_layer")

        # Determine stride and optional output padding for the transposed convolution.
        # This is necessary for certain stride values to ensure correct output dimensions.
        stride_tuple = stride if isinstance(stride, tuple) else (stride, stride)
        output_padding = (0, 1) if stride == (1, 2) else None

        # Depthwise transposed convolution layer performs the spatial upscaling,
        # grouped by the expanded channels. Followed by batch normalization and ReLU6.
        self.depthwise_layer_transpose = models.Sequential([
            layers.Conv2DTranspose(filters=in_channels * expansion,
                                   kernel_size=3,
                                   strides=stride_tuple,
                                   padding='same',
                                   output_padding=output_padding,
                                   groups=in_channels * expansion,
                                   use_bias=False,
                                   name=f"{name}_depthwise_conv_transpose"),
            layers.BatchNormalization(name=f"{name}_depthwise_bn"),
            layers.ReLU(max_value=6.0, name=f"{name}_depthwise_relu")
        ], name=f"{name}_depthwise_layer_transpose")

        # Pointwise transposed convolution layer (1x1 convolutions) reduces the channel
        # dimensions back to the desired number of output channels, followed by normalization.
        self.pointwise_layer_transpose = models.Sequential([
            layers.Conv2DTranspose(filters=out_channels, kernel_size=1, padding='same', use_bias=False, name=f"{name}_pointwise_conv_transpose"),
            layers.BatchNormalization(name=f"{name}_pointwise_bn")
        ], name=f"{name}_pointwise_layer_transpose")

    def call(self, inputs):
        """
        Forward pass of the InvertedResidualBlockTranspose layer.
        """
        # Pass inputs through the expansion layer
        x = self.expansion_layer(inputs)
        # Upscale and process through the depthwise transposed convolution
        x = self.depthwise_layer_transpose(x)
        # Reduce channel dimensions with the pointwise transposed convolution
        x = self.pointwise_layer_transpose(x)
        return x

def sampling(args):
    """
    Performs the reparameterization trick by sampling from a normal distribution with mean and log variance.
    """
    z_mean, z_log_var = args
    batch_size = K.shape(z_mean)[0]
    latent_dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch_size, latent_dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Learning rate schedule for the optimizer
initial_learning_rate = 1e-6
lr_schedule = ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=900,
    decay_rate=0.96,
    staircase=True)

def mobilenet_v2_decoder(is_time_distributed=False):
    """
    Builds a decoder model mimicking the MobileNetV2 architecture in reverse.
    """
    layer_type = "video" if is_time_distributed else "audio"
    # Define decoder layers, utilizing the custom InvertedResidualBlockTranspose for upsampling
    decoder_layers = [
        InvertedResidualBlockTranspose(320, 160, 6, 1, f'{layer_type}_irbt_1'),
        InvertedResidualBlockTranspose(160, 160, 6, 1, f'{layer_type}_irbt_2'),
        InvertedResidualBlockTranspose(160, 160, 6, 2, f'{layer_type}_irbt_3'),
        InvertedResidualBlockTranspose(160, 96, 6, 1, f'{layer_type}_irbt_4'),
        InvertedResidualBlockTranspose(96, 96, 6, 1, f'{layer_type}_irbt_5'),
        InvertedResidualBlockTranspose(96, 96, 6, 1, f'{layer_type}_irbt_6'),
        InvertedResidualBlockTranspose(96, 64, 6, 1, f'{layer_type}_irbt_7'),
        InvertedResidualBlockTranspose(64, 64, 6, 1, f'{layer_type}_irbt_8'),
        InvertedResidualBlockTranspose(64, 64, 6, 1, f'{layer_type}_irbt_9'),
        InvertedResidualBlockTranspose(64, 64, 6, 1, f'{layer_type}_irbt_10'),
        InvertedResidualBlockTranspose(64, 32, 6, 2, f'{layer_type}_irbt_11'),
        InvertedResidualBlockTranspose(32, 32, 6, 1, f'{layer_type}_irbt_12'),
        InvertedResidualBlockTranspose(32, 32, 6, 1, f'{layer_type}_irbt_13'),
        InvertedResidualBlockTranspose(32, 24, 6, 2, f'{layer_type}_irbt_14'),
        InvertedResidualBlockTranspose(24, 24, 6, 1, f'{layer_type}_irbt_15'),
        InvertedResidualBlockTranspose(24, 16, 6, 2, f'{layer_type}_irbt_16'),
        Conv2DTranspose(16, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False, name=f"{layer_type}_irbt_convt_17"),
        Conv2DTranspose(3, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, name=f"{layer_type}_irbt_convt_28"),
        Activation('tanh', name=f"{layer_type}_irbt_tanh_19")
    ]

    model_layers = []
    for i, layer in enumerate(decoder_layers):
        # Apply TimeDistributed wrapper if is_time_distributed is True
        if is_time_distributed:
            wrapped_layer = TimeDistributed(layer, name=f"{layer_type}_td_{i}")
            model_layers.append(wrapped_layer)
        else:
            model_layers.append(layer)

    return tf.keras.Sequential(model_layers)

# Instances for audio and video decoding using the MobileNetV2 inspired decoder
mobilenet_v2_decoder_audio = mobilenet_v2_decoder()
mobilenet_v2_decoder_video = mobilenet_v2_decoder(True)

class VAEModel:
    def __init__(self, audio_shape, video_shape, latent_dim=256):
        """
        Initializes the Variational Autoencoder (VAE) model with separate pathways for audio and video.
        """
        self.audio_shape = audio_shape
        self.video_shape = video_shape
        self.latent_dim = latent_dim

        print("Building model..")
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.vae = self.build_vae()


    def build_encoder(self):
        """
        Builds the encoder part of the VAE.
        """
        x_ = Input(shape=self.audio_shape, name='x_')
        y_ = Input(shape=self.video_shape, name='y_')

        # Audio encoder path - treating spectrogram as an image

        # Leveraging pretrained mobileNetV2 model for encoding 2d spectograms
        mobilnetv2_audio_encoder = applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=self.audio_shape)
        # Set all layers to not trainable
        for i, layer in enumerate(mobilnetv2_audio_encoder.layers):
            layer._name = 'audio_encoder_' + layer.name + f'_{i}'
            layer.trainable = False

        mobilnetv2_audio_encoder.layers[-1].trainable = True
        mobilnetv2_audio_encoder.layers[-2].trainable = True

        x = mobilnetv2_audio_encoder(x_)
        self.audio_encoded_shape = K.int_shape(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)


        # Video encoder path - time series frames
        # Leveraging pretrained mobileNetV2 model for encoding 2d spectograms
        mobilnetv2_video_encoder = applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=self.video_shape[1:])

        # Set all layers to not trainable
        for i, layer in enumerate(mobilnetv2_video_encoder.layers):
            layer._name = 'video_encoder_' + layer.name + f'_{i}'
            layer.trainable = False

        mobilnetv2_video_encoder.layers[-1].trainable = True

        y = TimeDistributed(mobilnetv2_video_encoder)(y_)
        self.video_encoded_shape = K.int_shape(y)
        y = TimeDistributed(Flatten())(y)
        y = LSTM(256)(y)
        y = Dense(256, activation='relu')(y)

        # Combine encoded features
        # attention_output = MultiHeadAttention(num_heads=2, key_dim=64)(
        #     query=y,  # Video features as queries
        #     value=x,  # Audio features as values
        #     key=x     # Audio features as keys
        # )

        # combined_encoded = Flatten()(attention_output)

        combined_encoded = Concatenate()([x, y])
        z_mean = Dense(self.latent_dim, name='z_mean')(combined_encoded)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(combined_encoded)
        z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])


        encoder = Model(inputs=[x_, y_], outputs=[z_mean, z_log_var, z], name='encoder')
        return encoder

    def build_decoder(self):
        """
        Builds the decoder part of the VAE.
        """
        # latent inputs  - unified latent space
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')

        # Audio decoder  - using InvertedResidualBlockTranspose layers
        x = Dense(np.prod(self.audio_encoded_shape[1:]), activation='relu')(latent_inputs)
        x = Reshape(self.audio_encoded_shape[1:])(x)
        x = mobilenet_v2_decoder_audio(x)

        # Video decoder  - using InvertedResidualBlockTranspose layers
        y = Dense(np.prod(self.video_encoded_shape[1:]), activation='relu')(latent_inputs)
        y = Reshape(self.video_encoded_shape[1:])(y)
        y = mobilenet_v2_decoder_video(y)

        # Decoder model
        decoder = Model(latent_inputs, [x, y], name='decoder')

        return decoder


    def build_vae(self):
        """
        Builds the VAE model.
        """
        x_ = Input(shape=self.audio_shape, name='encoded_audio')
        y_ = Input(shape=self.video_shape, name='encoded_video')

        z_mean, z_log_var, z = self.encoder([x_, y_])
        x, y = self.decoder(z)

        vae = Model(inputs=[x_, y_], outputs=[x, y], name='vae')


        # Audio is given more weights as we target to generate music from dance
        # The VAE can be also trained with different weights for generating video
        audio_weight = 1.0
        video_weight = 0.5
        kl_weight = 0.01

        # Reconstruction loss for each modalities using mean squared
        reconstruction_loss_audio = K.mean(K.square(x_ - x))
        reconstruction_loss_video = K.mean(K.square(y_ - y))
        reconstruction_loss = audio_weight * reconstruction_loss_audio + video_weight * reconstruction_loss_video

        kl_loss = kl_weight * -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        total_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(total_loss)

        vae.compile(optimizer=Adam(learning_rate=lr_schedule))

        return vae

    def get_latent_space_pair_generative(self, x_=[], y_=[]):
        """
        Returns the generative latent space pair for the given input.
        """
        _, _, z = self.encoder.predict([x_, y_])
        return self.decoder.predict(z)
