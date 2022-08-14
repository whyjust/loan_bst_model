#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   bst.py
@Time    :   2022/08/14 17:18:13
@Author  :   weiguang 
'''
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten
from base_feats import DenseFeat, SparseFeat, VarLenSparseFeat
from sequence import AttentionSequencePoolingLayer
from get_inputs_feats import *

def BST(dnn_feature_columns, history_feature_list, transformer_num=1, att_head_num=8,
        use_bn=False, dnn_hidden_units=(256, 128, 64), dnn_activation='relu', l2_reg_dnn=0,
        l2_reg_embedding=1e-6, dnn_dropout=0.0, seed=1024, task='binary'):
    features = build_input_features(dnn_feature_columns)
    input_list = list(features.values())

    use_behavior_length = features["seq_length"]
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)
    ) if dnn_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)
    ) if dnn_feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)
    ) if dnn_feature_columns else []

    history_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))

    # 不定长的稀疏特征
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)
    
    # 获取query hist与dnn对应的embedding矩阵
    embedding_dict = create_embedding_matrix(
        dnn_feature_columns, l2_reg_embedding, prefix="", seq_mask_zero=True)
    query_emb_list = embedding_lookup(
        embedding_dict, features, sparse_feature_columns, return_feat_list=history_feature_list, to_list=True)
    hist_emb_list = embedding_lookup(
        embedding_dict, features, history_feature_columns, return_feat_list=history_fc_names, to_list=True)
    dnn_input_emb_list = embedding_lookup(
        embedding_dict, features, sparse_feature_columns, mask_feat_list=history_feature_list, to_list=True)
    return query_emb_list, hist_emb_list, dnn_input_emb_list


if __name__ == "__main__":
    feature_columns = [
        SparseFeat('user', vocabulary_size=3, embedding_dim=10), 
        SparseFeat('gender', vocabulary_size=2, embedding_dim=4), 
        SparseFeat('item_id', vocabulary_size=3 + 1, embedding_dim=8),
        SparseFeat('cate_id', vocabulary_size=2 + 1, embedding_dim=4), 
        DenseFeat('pay_score', dimension=1)]
    feature_columns += [
        VarLenSparseFeat(
            SparseFeat('hist_item_id', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item_id'), 
            maxlen=4, length_name="seq_length"),
        VarLenSparseFeat(
            SparseFeat('hist_cate_id', 2 + 1, embedding_dim=4, embedding_name='cate_id'), 
            maxlen=4, length_name="seq_length")
    ]
    # Notice: History behavior sequence feature name must start with "hist_".
    behavior_feature_list = ["item_id", "cate_id"]
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    cate_id = np.array([1, 2, 2])  # 0 is mask value
    pay_score = np.array([0.1, 0.2, 0.3])
    # id与cate_id对应sequence序列
    hist_iid = np.array([[1, 2, 3, 0], [3, 2, 1, 0], [1, 2, 0, 0]])
    hist_cate_id = np.array([[1, 2, 2, 0], [2, 2, 1, 0], [1, 2, 0, 0]])
    # 序列的实际长度
    seq_length = np.array([3, 3, 2])
    feature_dict = {'user': uid, 'gender': ugender, 'item_id': iid, 'cate_id': cate_id,
                    'hist_item_id': hist_iid, 'hist_cate_id': hist_cate_id,
                    'pay_score': pay_score, 'seq_length': seq_length}
    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = np.array([1, 0, 1])

    query_emb_list, hist_emb_list, dnn_input_emb_list = BST(dnn_feature_columns=feature_columns, history_feature_list=behavior_feature_list, att_head_num=4)


