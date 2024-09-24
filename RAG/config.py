# -*- coding: utf-8 -*-
"""
@File:        config.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 07, 2024
@Description:
"""


"""--------------------模型相关配置 begin--------------------"""
# embedding model path
EMBEDDING_MODEL_PATH = "/Users/gene/project/tools/llm_tools/RAG/model_files/nlp_gte_sentence-embedding_chinese-base"
# re-rank model path
RERANK_MODEL_PATH = "/Users/gene/project/tools/llm_tools/RAG/model_files/models--BAAI--bge-reranker-v2-m3"

"""--------------------模型相关配置 end----------------------"""

"""--------------------日志相关配置 begin--------------------"""
LOGFILE_MAX_SIZE = 1024 * 1024 * 50  # 单个日志最大字节数
LOGFILE_BACKUP_NUMBER = 3  # 最大日志备份数
# gunicorn日志
loglevel = 'debug'  # 输出日志级别
pidfile = "logs/gunicorn.pid"  # 存放日志路径
accesslog = "logs/access.log"
errorlog = "logs/debug.log"
"""--------------------日志相关配置 end----------------------"""
# llm prompt 模板配置
PROMPT_TEMPLATE = """
你是一个问答机器人。你的任务是根据下述给定的已知信息回答用户的问题。确保你的回复完全依据下述已知信息。不能编造答案。如果下述已知信息不足以回答用户的问题，请直接回复“我无法回答您的问题”。
已知信息：
__INFO__
用户问：
__QUERY__
请用中文回答此问题。
"""
