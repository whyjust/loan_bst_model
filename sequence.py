#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   sequence.py
@Time    :   2022/08/07 16:16:15
@Author  :   weiguang 
'''
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import TruncatedNormal, Zeros, glorot_normal
from tensorflow.python.keras.layers import LSTM, Lambda, Layer
from tensorflow.python.keras.regularizers import l2

def reduce_mean(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None):
    try:
        return tf.reduce_mean(input_tensor, axis=axis, keep_dims=keep_dims, name=name, 
        reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_mean(input_tensor, axis=axis, keepdims=keep_dims, name=name)

def reduce_sum(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keep_dims, name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keep_dims, name=name)

def reduce_max(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None):
    try:
        return tf.reduce_max(input_tensor, axis=axis, keep_dims=keep_dims, name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_max(input_tensor, axis=axis, keepdims=keep_dims, name=name)

def div(x, y, name=None):
    try:
        return tf.div(x, y, name=name)
    except AttributeError:
        return tf.divide(x, y, name=name)

def softmax(logits, dim=-1, name=None):
    try:
        return tf.nn.softmax(logits, dim=dim, name=name)
    except TypeError:
        return tf.nn.softmax(logits, axis=dim, name=name)

class SequencePoolingLayer(Layer):
    """
    SequencePoolingLayer主要应用于可变序列特征的pooling操作(支持sum/mean/max)
    Input shape
    - A list of two  tensor [seq_value,seq_len]
    - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``
    - seq_len is a 2D tensor with shape : ``(batch_size, 1)``, indicate valid length of each sequence.

    Output shape
    - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.

    Arguments
    - mode: str. Pooling operation to be used,can be sum,mean or max.
    - supports_masking: If True,the input need to support masking.
    """
    def __init__(self, mode='mean', supports_masking=False, **kwargs):
        if mode not in ['sum', 'mean', 'max']:
            raise ValueError("mode must be sum or mean")
        self.mode = mode
        self.eps = tf.constant(1e-8, tf.float32)
        super(SequencePoolingLayer, self).__init__(**kwargs)
        self.supports_masking = supports_masking
    
    def build(self, input_shape):
        if not self.supports_masking:
            self.seq_len_max = int(input_shape[0][1])
        super(SequencePoolingLayer, self).build(input_shape)
    
    def call(self, seq_value_len_list, mask=None, **kwargs):
        if self.supports_masking:
            if mask is None:
                raise ValueError("When supports_masking=True,input must support masking!")
            uiseq_embed_list = seq_value_len_list
            mask = tf.cast(mask, tf.float32)
            user_behavior_length = reduce_sum(mask, axis=-1, keep_dims=True)
            mask = tf.expand_dims(mask, axis=2)
        else:
            uiseq_embed_list, user_behavior_length = seq_value_len_list
            # 根据user_behavior_length与seq_len_max输出对应的1/0矩阵
            mask = tf.sequence_mask(user_behavior_length, self.seq_len_max, dtype=tf.float32)
            mask = tf.transpose(mask, (0, 2, 1))
        embedding_size = uiseq_embed_list.shape[-1]
        mask = tf.tile(mask, [1, 1, embedding_size])

        if self.mode == "max":
            hist = uiseq_embed_list - (1 - mask) * 1e-9
            return reduce_max(hist, 1, keep_dims=True)
        hist = reduce_sum(uiseq_embed_list * mask, 1, keep_dims=False)
        if self.mode == "mean":
            hist = div(hist, tf.cast(user_behavior_length, tf.float32) + self.eps)
        hist = tf.expand_dims(hist, axis=1)
        return hist

    def compute_output_shape(self, input_shape):
        if self.supports_masking:
            return (None, 1, input_shape[-1])
        else:
            return (None, 1, input_shape[0][-1])
    
    def compute_mask(self, inputs, mask):
        return None
    
    def get_config(self):
        config = {'model': self.mode, 'supports_masking': self.supports_masking}
        base_config = super(SequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class LocalActivationUnit(Layer):
    """
    The LocalActivationUnit used in DIN with which the representation of
    user interests varies adaptively given different candidate items.

    Input shape
    - A list of two 3D tensor with shape:  ``(batch_size, 1, embedding_size)`` and ``(batch_size, T, embedding_size)``

    Output shape
    - 3D tensor with shape: ``(batch_size, T, 1)``.

    Arguments
    - **hidden_units**:list of positive integer, the attention net layer number and units in each layer.
    - **activation**: Activation function to use in attention net.
    - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix of attention net.
    - **dropout_rate**: float in [0,1). Fraction of the units to dropout in attention net.
    - **use_bn**: bool. Whether use BatchNormalization before activation or not in attention net.
    - **seed**: A Python integer to use as random seed.

    References
    - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """
    def __init__(self, hidden_units=(64, 32), activation="sigmoid", l2_reg=0, dropout_rate=0, \
        use_bn=False, seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.seed = seed
        self.supports_masking = True
        super(LocalActivationUnit, self).__init__(**kwargs)
    
    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `LocalActivationUnit` layer should be called '
                             'on a list of 2 inputs')

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError("Unexpected inputs dimensions %d and %d, expect to be 3 dimensions" % (
                len(input_shape[0]), len(input_shape[1])))

        if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1:
            raise ValueError('A `LocalActivationUnit` layer requires '
                             'inputs of a two inputs with shape (None,1,embedding_size) and (None,T,embedding_size)'
                             'Got different shapes: %s,%s' % (input_shape[0], input_shape[1]))
        size = 4 * int(input_shape[0][-1]) if len(self.hidden_units)==0 else self.hidden_units[-1]
        self.kernel = self.add_weight(shape=(size, 1), initializer=glorot_normal(seed=self.seed), name="kernel")
        self.bias = self.add_weight(shape=(1,), initializer=Zeros(), name="bias")
        self.dnn = DNN(self.hidden_units, self.activation, self.l2_reg, self.dropout_rate, self.use_bn, seed=self.seed)
        super(LocalActivationUnit, self).build(input_shape)
    
    def call(self, inputs, training=None, **kwargs):
        query, keys = inputs
        keys_len = keys.get_shape()[1]
        queries = K.repeat_elements(query, keys_len, 1)
        att_inputs = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)
        att_out = self.dnn(att_inputs, training=training)
        attention_score = tf.nn.bias_add(tf.tensordot(att_out, self.kernel, axes=(-1, 0)), self.bias)
        return attention_score
    
    def compute_output_shape(self, input_shape):
        return input_shape[1][:2] + (1,)

    def compute_mask(self, inputs, mask):
        return mask
    
    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'dropout_rate': self.dropout_rate, 'use_bn': self.use_bn, 'seed': self.seed}
        base_config = super(LocalActivationUnit, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WeightedSequenceLayer(Layer):
    """
    WeightedSequenceLayer主要应用于可变序列特征的weight scores
    Input shape
    - A list of two  tensor [seq_value,seq_len,seq_weight]
    - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``
    - seq_len is a 2D tensor with shape : ``(batch_size, 1)``, indicate valid length of each sequence.
    - seq_weight is a 3D tensor with shape: ``(batch_size, T, 1)``

    Output shape
    - 3D tensor with shape: ``(batch_size, T, embedding_size)``.

    Arguments
    - weight_normalization: bool.Whether normalize the weight score before applying to sequence.
    - supports_masking: If True,the input need to support masking.
    """
    def __init__(self, weight_normalization=True, supports_masking=False, **kwargs):
        super(WeightedSequenceLayer, self).__init__(**kwargs)
        self.weight_normalization = weight_normalization
        self.supports_masking = supports_masking
    
    def build(self, input_shape):
        if not self.supports_masking:
            self.seq_len_max = int(input_shape[0][1])
        super(WeightedSequenceLayer, self).build(input_shape)
    
    def call(self, input_list, mask=None, **kwargs):
        if self.supports_masking:
            if mask is None:
                raise ValueError("When supports_masking=True,input must support masking!")
            key_input, value_input = input_list
            mask = tf.expand_dims(mask[0], axis=2)
        else:
            key_input, key_length_input, value_input = input_list
            # 根据key_length_input与seq_len_max构建mask矩阵
            mask = tf.sequence_mask(key_length_input, self.seq_len_max, dtype=tf.bool)
            mask = tf.transpose(mask, (0, 2, 1))
        embedding_size = key_input.shape[-1]

        if self.weight_normalization:
            paddings = tf.ones_like(value_input) * (-2**32 + 1)
        else:
            paddings = tf.zeros_like(value_input)
        value_input = tf.where(mask, value_input, paddings)

        if self.weight_normalization:
            value_input = softmax(value_input, dim=1)
        
        if len(value_input.shape) == 2:
            value_input = tf.expand_dims(value_input, axis=2)
            # tile按照后续复制
            value_input = tf.tile(value_input, [1, 1, embedding_size])
        return tf.multiply(key_input, value_input)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, inputs, mask):
        if self.supports_masking:
            return mask[0]
        else:
            return None
    
    def get_config(self):
        config = {'weight_normalization': self.weight_normalization, 'supports_masking': self.supports_masking}
        base_config = super(WeightedSequenceLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AttentionSequencePoolingLayer(Layer):
    def __init__(self, att_hidden_units=(80, 40), att_activation="sigmoid", weight_normalization=False, 
                return_score=False, supports_masking=False, **kwargs):
        self.att_hidden_units = att_hidden_units
        self.att_activation = att_activation
        self.weight_normalization = weight_normalization
        self.return_score = return_score
        self.supports_masking = supports_masking
        super(AttentionSequencePoolingLayer, self.__init__(**kwargs))
    
    def build(self, input_shape):
        if not self.supports_masking:
            if not isinstance(input_shape, list) or len(input_shape) != 3:
                raise ValueError('A `AttentionSequencePoolingLayer` layer should be called on a list of 3 inputs!')
    
            if len(input_shape[0]) != 3 or len(input_shape[1]) != 3 or len(input_shape[2]) != 2:
                raise ValueError("Unexpected inputs dimensions,the 3 tensor dimensions are %d,%d and %d , expect to be 3,3 and 2" % (
                        len(input_shape[0]), len(input_shape[1]), len(input_shape[2])))

            if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1 or input_shape[2][1] != 1:
                raise ValueError('A `AttentionSequencePoolingLayer` layer requires inputs of a 3 tensor with shape (None,1,embedding_size),(None,T,embedding_size) and (None,1)'
                                 'Got different shapes: %s' % (input_shape))
