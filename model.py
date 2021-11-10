from tensorflow.keras.layers import *
from tensorflow.keras import models
from tensorflow.keras.activations import *
import tensorflow as tf
from tensorflow.python.keras.models import Sequential 
class MLP_Mixer():
    def __init__(self, inp_size = (224,224,3), classes = 2,  mixer_layer = 8, patch_size = 32, C = 512, Dc = 2048, Ds = 256, dropout = 0.0):
        # C: hidden size, Dc: hidden units of channel-mixing MLP, Ds: hidden units of token-mixing MLP
        self._inp_size = inp_size
        self._classes = classes
        self._mixerlayer = mixer_layer
        self._patch_size = patch_size
        self._C, self._Dc, self._Ds = (C, Dc, Ds)
        self._dropout = dropout
    def _token_(self, x):
        # input x.shpae : (None, Patch_size, Channel_size)
        _,P,_ = x.shape
        X = LayerNormalization()(x)
        X = tf.transpose(X, perm= [0,2,1]) # size of x : (None, Channel_size, Patch_size)
        X = Dense(units= self._Ds, use_bias= False)(X)
        X = tf.nn.gelu(X)
        X = Dense(units= P, use_bias= False)(X)
        X =  tf.transpose(X, perm= [0,2,1]) # size of x: (None, Patch_size, Channel_size)
        return add([x,X])
    def _channel_(self, x):
        # input x.shape: (None, Patch_size, Channel_size)
        X = LayerNormalization()(x)
        X = Dense(units= self._Dc, use_bias= False)(X)
        X = tf.nn.gelu(X)
        X =  Dense(units= self._C, use_bias= False)(X)
        return add([x,X])
    def _mixer_layer_(self, x):
        token = self._token_(x)
        channel = self._channel_(token)
        return channel
    def _patches_(self, image):
        patches = tf.image.extract_patches(
            images= image,
            sizes= [1, self._patch_size, self._patch_size, 1],
            strides= [1, self._patch_size, self._patch_size, 1],
            rates= [1,1,1,1],
            padding= 'VALID'
        )
        S = int((image.shape[1] ** 2) / self._patch_size ** 2)
        return  tf.reshape(patches, [tf.shape(image)[0], S, tf.shape(image)[-1] * self._patch_size ** 2])
    def build(self):
        # input image size (None, rows, cols, depths)  
        image = Input(shape= self._inp_size)
        X = self._patches_(image= image)
        # Per-patch Fully-connected 
        X = Dense(self._C)(X)
        # Mixer Layer 
        for i in range(self._mixerlayer):
            X = self._mixer_layer_(X)
        # Classification Layer
        X =  Sequential([
            GlobalAveragePooling1D(),
            Dropout(self._dropout),
            Dense(self._classes, activation= 'softmax')
        ])(X)
        return models.Model(image, X, name = 'mlp-mixer')


        



    