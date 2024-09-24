from typing import List

import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import RERANK_MODEL_PATH


class ReRank:
    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_PATH)
        self._model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL_PATH)
        self._model.eval()

    def rank(self, input_pairs: List[List[str]]) -> list:
        inputs = self._tokenizer(input_pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        outputs = self._model(**inputs, return_dict=True)
        logits = outputs.logits.view(-1, ).float()
        probabilities = F.softmax(logits, dim=0).data.tolist()
        sorted_list = sorted(zip(probabilities, input_pairs), key=lambda x: x[0], reverse=True)
        return sorted_list


if __name__ == '__main__':
    import time
    import threading

    rk = ReRank()
    start_time = time.time()
    input_str_list = [
        ['what is panda?', 'hi'],
        ['what is panda?',
         'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.'],
        ['柏林有多少人?', '柏林的面积为891.82平方公里，登记人口为3520031人。'],
        ['柏林有多少人?', '柏林以其博物馆而闻名。每天有500人进出。'],
    ]
    probs = rk.rank(input_str_list)
    print("spend time: ", time.time() - start_time)
    print(probs)
