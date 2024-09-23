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
    NORMAL = "chat"
    LINK = "link"
    REPORT = "report"


class ModelItem:
    """模型配置"""
    LLM = "llama3-dpo"
    EMBEDDING = ""


class LLMConfig:
    """生成模型配置"""
    TOP_P = 0.7
    TEMPERATURE = 0.8
