# -*- coding: utf-8 -*-
"""
@File:        embedding.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 19, 2024
@Description:
"""
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from config import EMBEDDING_MODEL_PATH


class Embedding:
    def __init__(self,
                 model_path=EMBEDDING_MODEL_PATH
                 ):
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self._model.eval()

    def embedding_text(self, inputs: List[str]) -> torch.Tensor:
        # Tokenize the input texts
        batch_dict = self._tokenizer(inputs, max_length=8192, padding=True, truncation=True, return_tensors='pt')
        outputs = self._model(**batch_dict)
        embeddings = outputs.last_hidden_state[:, 0]
        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        return embeddings


if __name__ == '__main__':
    input_texts = [
        "what is the capital of China?",
        "how to implement quick sort in python?",
        "Beijing",
        "sorting algorithms"
    ]
    inputs = ["吃完海鲜可以喝牛奶吗?",
              "不可以，早晨喝牛奶不科学",
              "不可以，嘻嘻嘻嘻嘻嘻呃呃呃邵丹单方面防腐剂钓底打开发",
              "本模型基于Dureader Retrieval中文数据集(通用领域)上训练，在垂类领域英文文本上的文本效果会有降低，请用户自行评测后决定如何使用",
              "海鲜默认向量维度768, 牛奶scores中的score计算两个向量之间的内成积距离得到",
              "吃了海鲜后是不能再喝牛奶的，因为牛奶中含得有维生素C，如果海鲜喝牛奶一起服用会对人体造成一定的伤害",
              "吃海鲜是不能同时喝牛奶吃水果，这个至少间隔6小时以上才可以。",
              "吃海鲜是不可以吃柠檬的因为其中的维生素C会和海鲜中的矿物质形成砷"]

    emb = Embedding()
    print(emb.embedding_text(inputs))
