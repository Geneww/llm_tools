# -*- coding: utf-8 -*-
"""
@File:        chinese_utils.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 07, 2024
@Description:
    中文搜索句子切分检索关键词
"""
import re
import jieba
import nltk
from nltk.corpus import stopwords


# nltk.download('stopwords')


def sentence_to_keywords(input_string):
    """
    将句子转换成检索关键词
    :param input_string:
    :return:
    """
    word_tokens = jieba.cut_for_search(input_string)
    print(word_tokens)
    stop_words = set(stopwords.words('chinese'))
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    print(filtered_sentence)
    return ' '.join(filtered_sentence)


def sent_tokenize(input_string):
    """
    按照标点进行断句
    :param input_string:
    :return:
    """
    sentences = re.split(r'(?<=[。！？；?!])', input_string)
    # 去掉空字符串
    return [sentence for sentence in sentences if sentence.strip()]


if __name__ == '__main__':
    print(sentence_to_keywords("RAG模型部署源码资料已经打包准备好了"))
    print(sent_tokenize("ok, i'm fine, 你在干嘛啊？我在吃饭。"))
