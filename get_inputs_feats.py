#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   get_inputs_feats.py
@Time    :   2022/08/07 16:15:49
@Author  :   weiguang 
'''
from unicodedata import lookup
import tensorflow as tf
from itertools import chain
from tensorflow.python.keras.layers import Input, Lambda, Embedding
from tensorflow.python.keras.regularizers import l2
from collections import OrderedDict, defaultdict
from base_feats import DenseFeat, SparseFeat, VarLenSparseFeat

def build_input_features(feature_columns, prefix=""):
    """
    将feature_columns中按照SparseFeat/DenseFeat/VarLenSparseFeat进行分类定义Input
    Args:
        feature_columns (_type_): 特征列
        prefix (str, optional): 前缀. Defaults to "".

    Raises:
        TypeError: 必须为3中定义类型特征

    Returns:
        _type_: 特征字典
    """
    input_features= OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            input_features[fc.name] = Input(shape=(1, ), name=prefix+fc.name, dtype=fc.dtype)
        elif isinstance(fc, DenseFeat):
            input_features[fc.name] = Input(shape=(fc.dimension, ), name=prefix+fc.name, dtype=fc.dtype)
        elif isinstance(fc, VarLenSparseFeat):
            input_features[fc.name] = Input(shape=(fc.maxlen, ), name=prefix+fc.name, dtype=fc.dtype)
            if fc.weight_name is not None:
                input_features[fc.weight_name] = Input(shape=(fc.maxlen, 1), name=prefix+fc.weight_name, dtype="float32")
            if fc.length_name is not None:
                input_features[fc.length_name] = Input(shape=(1,), name=prefix+fc.length_name, dtype="int32")
        else:
            raise TypeError("Invalid feature column type, got", type(fc))
    return input_features

def get_feature_names(feature_columns):
    """
    获取特征列表
    Args:
        feature_columns (_type_): 特征列

    Returns:
        _type_: 特征列表
    """
    features = build_input_features(feature_columns)
    return list(features.keys())

def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, l2_reg, prefix='sparse_', seq_mask_zero=True):
    """
    构建embedding字典
    Args:
        sparse_feature_columns (_type_): 稀疏特征列表
        varlen_sparse_feature_columns (_type_): 序列特征
        l2_reg (_type_): l2正则
        prefix (str, optional): 前缀. Defaults to 'sparse_'.
        seq_mask_zero (bool, optional): 序列是否mask零. Defaults to True.

    Returns:
        _type_: 稀疏特征字典
    """
    sparse_embedding = {}
    for feat in sparse_feature_columns:
        emb = Embedding(feat.vocabulary_size, feat.embedding_dim, 
                        embeddings_initializer=feat.embeddings_initializer,
                        embeddings_regularizer=l2(l2_reg),
                        name=prefix+'_emb_'+feat.embedding_name)
        emb.trainable = feat.trainable
        sparse_embedding[feat.embedding_name] = emb
    
    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                            embeddings_initializer=feat.embeddings_initializer,
                            embeddings_regularizer=l2(l2_reg),
                            name=prefix+'_seq_emb_'+feat.name,
                            mask_zero=seq_mask_zero)
            emb.trainable = feat.trainable
            sparse_embedding[feat.embedding_name] = emb
    return sparse_embedding

def create_embedding_matrix(feature_columns, l2_reg, prefix="", seq_mask_zero=True):
    """
    创建embedding字典
    Args:
        feature_columns (_type_): 特征列表
        l2_reg (_type_): 正则项
        prefix (str, optional): 前缀. Defaults to "".
        seq_mask_zero (bool, optional): 序列是否mask零. Defaults to True.

    Returns:
        _type_: _description_
    """
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns) if feature_columns else []
    )
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns) if feature_columns else []
    )
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, 
                                            l2_reg, prefix=prefix+'sparse', seq_mask_zero=seq_mask_zero)
    return sparse_emb_dict

def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, reture_feat_list=(), to_list=False):
    """
    获取sparse-embedding
    Args:
        sparse_embedding_dict (_type_): sparse特征embedding字典
        sparse_input_dict (_type_): sparse输入字典
        sparse_feature_columns (_type_): 稀疏特征列表
        reture_feat_list (tuple, optional): 返回特征列表. Defaults to ().
        to_list (bool, optional): 是否转list. Defaults to False.

    Returns:
        _type_: 返回sparse embedding dict
    """
    group_embedding_dict = defaultdict(list)
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if (len(reture_feat_list) == 0 or feature_name in reture_feat_list):
            lookup_idx = sparse_input_dict[feature_name]
        
        group_embedding_dict[fc.group_name].append(sparse_embedding_dict[embedding_name](lookup_idx))
    if to_list:
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict

def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    """
    获取varlen-embedding列表
    Args:
        embedding_dict (_type_): embedding字典
        sequence_input_dict (_type_): 序列输入字典
        varlen_sparse_feature_columns (_type_): varlen-sparse特征列表

    Returns:
        _type_: 返回var-embedding-vec字典
    """
    var_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        lookup_idx = sequence_input_dict[feature_name]
        var_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)
    return var_embedding_vec_dict


def get_varlen_pooling_list(embedding_dict, features, varlen_sparse_feature_columns, tolist=False):
    pooling_vec_list = defaultdict(list)
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        combiner = fc.combiner
        feature_length_name = fc.length_name

        if feature_length_name is not None:
            if fc.weight_name is not None:
                pass