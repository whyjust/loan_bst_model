#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   base_feats.py
@Time    :   2022/08/07 16:15:32
@Author  :   weiguang 
'''
from collections import namedtuple
from tensorflow.python.keras.initializers import RandomNormal

DEFAULT_GROUP_NAME = "default_group"

class SparseFeat(namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim', \
    'use_hash', 'vocabulary_path', 'dtype', 'embeddings_initializer', 'embedding_name', \
    'group_name', 'trainable'])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, vocabulary_path=None, \
        dtype="int32", embeddings_initializer=None, embedding_name=None, group_name=DEFAULT_GROUP_NAME, trainable=True):
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if embeddings_initializer is None:
            embeddings_initializer = RandomNormal(mean=0.0, steddev=0.0001, seed=2020)
        
        if embedding_name is None:
            embedding_name = name
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, vocabulary_path, dtype,\
            embeddings_initializer, embedding_name, group_name, trainable)
    
    def __hash__(self):
        return self.name.__hash__()

class VarLenSparseFeat(namedtuple('VarLenSparseFeat', ['sparsefeat', 'maxlen', 'combiner', 'length_name', 'weight_name', 'weight_norm'])):
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner="mean", length_name=None, weight_name=None, weight_norm=True):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name, weight_name, weight_norm)
    
    @property
    def name(self):
        return self.sparsefeat.name
    
    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size
    
    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim
    
    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def vocabulary_path(self):
        return self.sparsefeat.vocabulary_path

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embeddings_initializer(self):
        return self.sparsefeat.embeddings_initializer

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    @property
    def trainable(self):
        return self.sparsefeat.trainable

    def __hash__(self):
        return self.name.__hash__()
    

class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype', 'transform_fn'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32", transform_fn=None):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype, transform_fn)
    
    def __hash__(self):
        return self.name.__hash__()
    
