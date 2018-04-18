# -*- coding:utf-8 -*-
"""
基础的向量操作，以及向量化模块
"""
from __future__ import print_function
import jieba
import numpy as np
from bot.nlp.segment.mix_segment import segment
from bot.utils.time_wrapper import timing


# @timing
def vec_sentence(sentence, w2v_model, exclude_words, word_dist,
                 interpolation=0, interpolation_value=20, cut_all=True,
                 return_word_group=False, verbose=1):
    """
    使用 tf-idf 加权的 Word2Vec 模型对输入的句子做向量化表示
    :param sentence: sentence after word cut
    :param w2v_model: gensim 训练的 Word2Vec 模型
    :param exclude_words: list_of_words, 排除词列表, 列表中的单词将不被计算入句子的向量中
    :param word_dist:
        word distribution, dict(key=word, value=idf_value)
    :param interpolation: 如何处理不在 word_dist 中的单词(OOV: Out Of Vocabulary)
        0 --> init with 20.0, use for idfs
        1 --> init with 0.0, use for a distribution
        2 --> init with 1.0, use for a distribution
        3 --> init with "interpolation_value"
    :param interpolation_value: 当 interpolation == 3 时，用来初始化 OOV 的值
    :param cut_all: bool
    :param return_word_group: bool
    :param verbose: int
    :return:
        return_word_group == False:
            numpy-array-1D: vector of sentence
        return_word_group == True:
            numpy-array-2D: 2D-vector of words in sentence
    """
    if not isinstance(sentence, list):
        sentence = jieba.lcut(sentence)
        # sentence = segment(
        #     sentence, char_or_word="word",
        #     only_jieba=True, cut_all=False
        # )
    vector_size = w2v_model.vector_size
    cur_vec = []
    dfl = []
    for w in sentence:
        if w in exclude_words:
            continue
        if w not in word_dist:
            if interpolation == 0:
                dfw = 20.0
            elif interpolation == 1:
                dfw = 0.0
            elif interpolation == 2:
                dfw = 1.0
            else:
                dfw = interpolation_value
        else:
            dfw = word_dist[w]

        if w in w2v_model:
            if cut_all:
                w4v = []
                # wl = jieba.cut(w, cut_all=True)
                wl = segment(
                    w, char_or_word="word",
                    only_jieba=True, cut_all=True
                )
                for ww in wl:
                    if ww in w2v_model:
                        w4v.append(ww)
                if len(w4v) == 0:
                    continue
                w4v = w2v_model[w4v].reshape(-1, vector_size)
                cv = np.mean(w4v, axis=0)
            else:
                cv = w2v_model[w]
        else:
            w4v = []
            # wl = jieba.cut(w, cut_all=True)
            wl = segment(
                w, char_or_word="word",
                only_jieba=True, cut_all=True
            )
            for ww in wl:
                if ww in w2v_model:
                    w4v.append(ww)
            if len(w4v) == 0:
                continue
            w4v = w2v_model[w4v].reshape(-1, vector_size)
            cv = np.mean(w4v, axis=0)

        cur_vec.append(cv)
        dfl.append(dfw)

    if len(cur_vec) == 0:
        if verbose >= 1:
            print("[WARNING] Cannot vec sentence: %s" % (u"".join(sentence)).encode("utf-8"))
        return None

    # 转化为 numpy-like
    cur_vec = np.array(cur_vec)
    dfl_array = np.array(dfl, dtype=np.float)

    # 计算每个单词的分布
    sum_dfl = np.sum(dfl_array, dtype=np.float)
    pdf = dfl_array / sum_dfl

    # 用分布加权句子中各个单词的向量
    assert pdf.shape[0] == cur_vec.shape[0]
    cur_vec = cur_vec * pdf.repeat(cur_vec.shape[1]).reshape(cur_vec.shape)

    if return_word_group:
        return cur_vec
    else:
        return np.sum(cur_vec, axis=0).reshape(-1)


def unit_vec(vec, norm='l2', axis=0):
    """
    :param vec: numpy array
    :param norm:
    :param axis:
    :return:
    """
    if norm == 'l1':
        vec_len = np.sum(np.abs(vec), axis=axis)
    elif norm == 'l2':
        vec_len = np.sqrt(np.sum(np.power(vec, 2), axis=axis))
    else:
        raise ValueError()

    if np.prod(vec_len, axis=0) > 0.0:
        return np.transpose(np.transpose(vec) / vec_len)
    else:
        return vec
