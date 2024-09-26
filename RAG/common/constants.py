# -*- coding: utf-8 -*-
"""
@File:        constants.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 19, 2024
@Description:
"""


class QueryType:
    """大模型类型"""
    DOC = "document"
    NORMAL = "common"
    LINK = "link"
    REPORT = "report"


class ResponseType:
    """大模型回复类型"""
    STREAM = "stream"  # 流式返回
    BLOCK = "block"  # 一次性返回


class ModelItem:
    """模型配置"""
    LLM = "llama3-dpo"
    EMBEDDING = ""


class LLMConfig:
    """生成模型配置"""
    TOP_P = 0.7
    TEMPERATURE = 0.8
