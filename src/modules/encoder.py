import tensorflow as tf
from tensorflow import nn
from tensorflow.keras import Sequential
from tensorflow.models import resnet18, resnet50, resnet101, resnet152, vgg16, vgg19, inception_v3
import random
import numpy as np

class EncoderCNN(tf.keras.layers.Layer):
    def __init__(self, embed_size, dropout=0.5, image_model='resnet101', pretrained=True):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = globals()[image_model](pretrained=pretrained)
        modules = list(resnet.children())[:-2]  # delete the last fc layer.
        self.resnet = Sequential(*modules)
        # Original pytorch layers
        # nn.Conv2D(resnet.fc.in_features, embed_size, kernel_size=1, padding=0)
        # nn.Dropout2d(dropout)
        self.linear = Sequential([tf.keras.layers.Conv2D(embed_size, kernel_size=1, padding='valid'),
                                  tf.keras.layers.Dropout(dropout)])

    # Original pytorch
    # def forward(self, images, keep_cnn_gradients=False):
    #     """Extract feature vectors from input images."""
    #     if keep_cnn_gradients:
    #         raw_conv_feats = self.resnet(images)
    #     else:
    #         with torch.no_grad():
    #             raw_conv_feats = self.resnet(images)
    #     features = self.linear(raw_conv_feats)
    #     features = features.view(features.size(0), features.size(1), -1)
    #     return features

    def call(self, images, keep_cnn_gradients=False):
        if keep_cnn_gradients:
            raw_conv_feats = self.resnet(images)
        else:
            raw_conv_feats = tf.stop_gradient(self.resnet(images))
        features = self.linear(raw_conv_feats)
        features = tf.reshape(features, [features.size[0], features.size[1], -1])
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
        # self.pad_value = num_classes - 1
        self.pad_value = 0
        self.linear = embeddinglayer
        self.dropout = dropout
        self.embed_size = embed_size

    def call(self, x, onehot_flag=False):
        if onehot_flag:
            embeddings = tf.matmul(x, self.linear.weights[0]) # may need to transpose weight matrix
        else:
            embeddings = self.linear(x)
        # not sure if should switch to using a dropout layer instead
        embeddings = tf.nn.dropout(embeddings, self.dropout)
        embeddings = tf.transpose(embeddings, perm=[0, 2, 1])

        return embeddings