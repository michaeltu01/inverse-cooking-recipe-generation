import tensorflow as tf
from tensorflow import nn
from keras import Sequential
from keras.applications import ResNet101
# vgg16, vgg19, inception_v3
import random
import numpy as np

class EncoderCNN(tf.keras.layers.Layer):
    def __init__(self, embed_size, dropout=0.5, image_model='resnet101', pretrained=True):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        self.resnet = ResNet101(weights='imagenet' if pretrained else None, include_top=False)
        # modules = list(self.resnet.children())[:-2]  # delete the last fc layer.
        for layer in self.resnet.layers:
            layer.trainable = False

        self.resnet = tf.keras.Model(inputs=self.resnet.input, outputs=self.resnet.layers[-3].output)
        # self.resnet = Sequential(*modules)
        self.linear = Sequential([tf.keras.layers.Conv2D(embed_size, kernel_size=1, padding='valid'),
                                  tf.keras.layers.Dropout(dropout)])

    def call(self, images, keep_cnn_gradients=False):
        if keep_cnn_gradients:
            raw_conv_feats = self.resnet(images)
        else:
            raw_conv_feats = tf.stop_gradient(self.resnet(images))
        features = self.linear(raw_conv_feats)
        features = tf.reshape(features, [tf.shape(features)[0], tf.shape(features)[1], -1])
        return features


class EncoderLabels(tf.keras.layers.Layer):
    def __init__(self, embed_size, num_classes, dropout=0.5, embed_weights=None, scale_grad=False):
        super(EncoderLabels, self).__init__()
        # embeddinglayer = nn.Embedding(num_classes, embed_size, padding_idx=num_classes-1, scale_grad_by_freq=scale_grad)
        # TODO: need to pad and somehow pass the last pytorch argument
        if embed_weights is not None:
            embeddinglayer = tf.keras.layers.Embedding(num_classes, embed_size, mask_zero=True, weights=embed_weights)
        else:
            embeddinglayer = tf.keras.layers.Embedding(num_classes, embed_size, mask_zero=True)
        self.pad_value = num_classes - 1
        # self.pad_value = 0
        self.linear = embeddinglayer
        self.dropout = dropout
        self.embed_size = embed_size

    def call(self, x, onehot_flag=False, training=True):
        if onehot_flag:
            embeddings = tf.matmul(x, self.linear.weights[0]) # may need to transpose weight matrix
        else:
            embeddings = self.linear(x)
        dropout = tf.keras.layers.Dropout(self.dropout)
        embeddings = dropout(embeddings, training=training)
        embeddings = tf.transpose(embeddings, perm=[0, 2, 1])

        return embeddings