# -*- coding:utf-8 -*-
from __future__ import print_function
import os
import codecs
import gensim
import logging

from bot.nlp.segment.mix_segment import segment


class Word2VecLight(object):
    def __init__(self, w2v_model_pth):
        w2v = load_w2v(w2v_model_pth)
        self.vocab = dict()
        self.matrix = w2v.wv.syn0
        for w, v in w2v.wv.vocab.items():
            self.vocab[w] = v.index
        del w2v

    def __contains__(self, item):
        if item in self.vocab:
            return True
        else:
            return False

    def __getitem__(self, item):
        if item not in self.vocab:
            return None
        else:
            return self.matrix[self.vocab[item]]


def line_generator(file_name, save_file=None, cut_sentence=True, segment_excludes=None, max_cnt=None, only_jieba=False):
    """
    :param file_name: 语料文件名
    :param save_file: 文件名，指定后保存分词后的结果到该文件下，None 表示不保存，默认为 None
    :param cut_sentence: True 返回分词后的向量，False 返回句子 unicode，默认为 True
    :param segment_excludes: 对应 mix_segment 模块中的 sub_cut_exclude
    :param max_cnt:
    :param only_jieba:
    :return:
    """
    if save_file:
        with codecs.open(save_file, 'w', encoding="utf-8"):
            pass

    if not isinstance(file_name, (list, tuple)):
        # 从文件中获取语料
        print("[INFO] line_generator from file.")
        with codecs.open(file_name, 'r', encoding="utf-8") as fp:
            cnt = 0
            while 1:
                if cnt % 1000 == 0:
                    print("line number:", cnt)
                if max_cnt and cnt > max_cnt:
                    break
                line = fp.readline()
                if line is None or len(line) == 0:
                    break
                line = line.strip()
                if len(line) == 0:
                    continue
                if not isinstance(line, unicode):
                    line = line.decode("utf-8")

                if cut_sentence:
                    word_list = segment(line, sub_cut_exclude=segment_excludes, only_jieba=only_jieba)
                    cnt += 1
                    if save_file:
                        with codecs.open(save_file, 'a', encoding="utf-8") as fp_w:
                            fp_w.write(u" ".join(word_list) + u"\n")
                    yield word_list
                else:
                    yield line
    else:
        if isinstance(file_name[0], (str, unicode)):
            # 未分词
            print("[INFO] line_generator from list of un-word-cut.")
            cnt = 0
            for line in file_name:
                if cnt % 1000 == 0:
                    print("line number:", cnt)
                if max_cnt and cnt > max_cnt:
                    break
                if line is None or len(line) == 0:
                    break
                line = line.strip()
                if len(line) == 0:
                    continue
                if not isinstance(line, unicode):
                    line = line.decode("utf-8")

                if cut_sentence:
                    word_list = segment(line, sub_cut_exclude=segment_excludes, only_jieba=only_jieba)
                    cnt += 1
                    if save_file:
                        with codecs.open(save_file, 'a', encoding="utf-8") as fp_w:
                            fp_w.write(u" ".join(word_list) + u"\n")
                    yield word_list
                else:
                    cnt += 1
                    yield line
        else:
            # 已分词
            print("[INFO] line_generator from list of word-cut.")
            cnt = 0
            for line in file_name:
                if cnt % 1000 == 0:
                    print("line number:", cnt)
                if max_cnt and cnt > max_cnt:
                    break
                cnt += 1
                yield line


def load_line_gen(corpus_file):
    """
    读取分词后的语料
    :param corpus_file: 语料文件名
    :return:
    """
    with codecs.open(corpus_file, 'r', encoding="utf-8") as fp:
        cnt = 0
        while 1:
            if cnt % 1000 == 0:
                print("line number:", cnt)
            line = fp.readline()
            if line is None or len(line) == 0:
                break
            line = line.strip()
            if len(line) == 0:
                continue
            word_list = line.split(u" ")
            cnt += 1
            yield word_list


def train_w2v(file_name, model_path, save_split_corpus=True, cut_with="tokenizer",
              params=None, max_cnt=None):
    """
    首次训练模型
    :param file_name: string, 训练集所在的文件名，其中每一行都是一个完整的句子
    :param model_path: string, 模型保存的路径
    :param save_split_corpus: 是否保存训练集切好词的结果
    :param cut_with: str, tokenizer | space, 选择分词的方式
    :param params: dict，存放训练模型需要的参数，默认为 None
        params={
            "segment_excludes"  : None,
            "min_count"         : 0,
            "iter"              : 100,
            "dimension"         : 128,
            "only_jieba"        : False,
            "sg"                : 1,
            "negative"          : 5,
        }
    :param max_cnt:
    :return:
    """
    # default
    segment_excludes = None
    only_jieba = False
    cur_min_count = 0
    cur_iter = 100
    cur_sg = 1
    cur_negative = 5
    cur_dim = 128
    workers = 4

    if params:
        only_jieba      = params["only_jieba"] if "only_jieba" in params else only_jieba
        cur_min_count   = params["min_count"] if "min_count" in params else cur_min_count
        cur_iter        = params["iter"] if "iter" in params else cur_iter
        cur_sg          = params["sg"] if "sg" in params else cur_sg
        cur_negative    = params["negative"] if "negative" in params else cur_negative
        cur_dim         = params["dimension"] if "dimension" in params else cur_dim
        workers         = params["workers"] if "workers" in params else workers
        segment_excludes = params["segment_excludes"] if "segment_excludes" in params else segment_excludes

    print("*" * 40)
    print("Training...")

    # read file
    if cut_with == "space":
        sent_gen = list(load_line_gen(corpus_file=file_name))
    else:
        if not isinstance(file_name, (list, tuple)):
            file_suffix = "_cut.".join(file_name.split("/")[-1].split("."))
            if save_split_corpus:
                sent_gen = list(line_generator(
                    file_name,
                    save_file=os.path.join(model_path, file_suffix),
                    segment_excludes=segment_excludes,
                    max_cnt=max_cnt,
                    only_jieba=only_jieba
                ))
            else:
                sent_gen = list(line_generator(
                    file_name,
                    save_file=None,
                    segment_excludes=segment_excludes,
                    max_cnt=max_cnt,
                    only_jieba=only_jieba
                ))
        else:
            if save_split_corpus:
                sent_gen = list(line_generator(
                    file_name,
                    save_file=os.path.join(model_path, "corpus_cut.txt"),
                    segment_excludes=segment_excludes,
                    max_cnt=max_cnt,
                    only_jieba=only_jieba
                ))
            else:
                sent_gen = list(line_generator(
                    file_name,
                    save_file=None,
                    segment_excludes=segment_excludes,
                    max_cnt=max_cnt,
                    only_jieba=only_jieba
                ))

    # train and save
    model = gensim.models.Word2Vec(
        sent_gen, min_count=cur_min_count, iter=cur_iter,
        size=cur_dim, sg=cur_sg, negative=cur_negative, workers=workers
    )

    if model_path.endswith(".w2v"):
        model_file = model_path
    else:
        model_file = os.path.join(model_path, "model.w2v")
    # model.save_word2vec_format(model_file)
    model.save(model_file)
    print("Model write completed! Model file is:", model_file)
    return model


def update_w2v(file_name, model_path, update_and_save=False,
               save_split_corpus=True, cut_with="tokenizer",
               params=None, max_cnt=None, mmap=False):
    """
    在已有模型的基础上再训练
    :param file_name: str|list: str for corpus file name, list for iterable corpus.
    :param model_path:
    :param update_and_save:
    :param save_split_corpus:
    :param cut_with: str, tokenizer | space, 选择分词的方式
    :param params: dict，存放训练模型需要的参数，默认为 None
        params={
            "segment_excludes"  : None,
            "min_count"         : 0,
            "iter"              : 100,
            "only_jieba"        : False,
        }
    :param max_cnt:
    :param mmap:
    :return:
    """
    only_jieba = False
    segment_excludes = None
    workers = 4
    if params:
        segment_excludes = params["segment_excludes"] if "segment_excludes" in params else segment_excludes
        only_jieba = params["only_jieba"] if "only_jieba" in params else only_jieba
        workers = params["workers"] if "workers" in params else workers

    print("*" * 40)
    print("Updating...")

    if cut_with == "space":
        sent_gen = list(load_line_gen(corpus_file=file_name))
    else:
        if not isinstance(file_name, (list, tuple)):
            file_suffix = file_name.split("/")[-1]
            if save_split_corpus:
                sent_gen = list(line_generator(
                    file_name,
                    save_file=os.path.join(model_path, file_suffix),
                    segment_excludes=segment_excludes,
                    max_cnt=max_cnt,
                    only_jieba=only_jieba
                ))
            else:
                sent_gen = list(line_generator(
                    file_name,
                    save_file=None,
                    segment_excludes=segment_excludes,
                    max_cnt=max_cnt,
                    only_jieba=only_jieba
                ))
        else:
            sent_gen = list(line_generator(
                file_name,
                save_file=None,
                segment_excludes=segment_excludes,
                max_cnt=max_cnt,
                only_jieba=only_jieba
            ))

    if model_path.endswith(".w2v"):
        model_file = model_path
    else:
        model_file = os.path.join(model_path, "model.w2v")

    model = load_w2v(model_path, mmap)
    cur_min_count = model.min_count
    cur_min_count = params["min_count"] if "min_count" in params else cur_min_count

    if "iter" in params:
        cur_iter = params["iter"]
    else:
        cur_iter = model.iter

    model.min_count = cur_min_count

    # try:
    #     model.build_vocab(sent_gen, update=True)
    #     model.train(
    #         sent_gen, total_examples=model.corpus_count, epochs=cur_iter,
    #         start_alpha=model.alpha, end_alpha=model.min_alpha)
    # except Exception as e:
    # # 考虑版本兼容性，gensim 3.+ 存在 load 模型后成员不完整的问题
    # logging.warning("[update_w2v] %s, %s" %
    #                 (
    #                     str(e),
    #                     "考虑版本兼容性，gensim 3.+ 存在 load 模型后成员不完整的问题"
    #                 ))
    # temp_model = gensim.models.Word2Vec()
    # for attr_name, att_value in model.__dict__.items():
    #     temp_model.__setattr__(attr_name, att_value)
    # model = temp_model
    # del temp_model
    model.workers = workers
    model.build_vocab(sent_gen, update=True)
    model.train(
        sent_gen, total_examples=model.corpus_count, epochs=cur_iter,
        start_alpha=model.alpha, end_alpha=model.min_alpha)

    if update_and_save:
        # model.wv.save_word2vec_format(model_file)
        model.save(model_file)

    print("[INFO] Model update completed! [File] ", model_file, " [UpdateSave]", update_and_save)
    return model


def load_w2v(model_path, mmap='r'):
    """
    加载 w2v 模型
    :param model_path: 模型保存的路径, 或者文件名
    :param mmap:
        word2vec 内存映射的调用方式，节省word2vec内存占用（使用了 numpy 的 memmap 功能）
        The file is opened in this mode:
        ‘r’ : Open existing file for reading only.
        ‘r+’: Open existing file for reading and writing.
        ‘w+’: Create or overwrite existing file for reading and writing.
        ‘c’ : Copy-on-write: assignments affect data in memory, but changes are not saved to disk.
            The file on disk is read-only.
        default is None
    :return:
    """
    if model_path.endswith(".w2v"):
        model_file = model_path
    else:
        model_file = os.path.join(model_path, "model.w2v")
    model = gensim.models.Word2Vec.load(model_file, mmap=mmap)
    # model = gensim.models.Word2Vec().wv.load_word2vec_format(model_file)
    # model = model.load(model_file)

    temp_model = gensim.models.Word2Vec()
    for attr_name, att_value in model.__dict__.items():
        if attr_name in temp_model.__dict__:
            temp_model.__setattr__(attr_name, att_value)
    model = temp_model
    del temp_model

    return model






