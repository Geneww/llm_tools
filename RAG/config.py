# -*- coding: utf-8 -*-
"""
@File:        config.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 07, 2024
@Description:
"""

"""--------------------模型相关配置 begin--------------------"""
# LLM model
LLM_MODEL = "llama3-simple"
LLM_MODEL_URL = "http://192.168.124.100:11434"
LLM_MODEL_TOP_P = 0.8
LLM_MODEL_TEMPERATURE = 0.8
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

RAG_PROMPT_TEMPLATE = [
    {"role": "system",
     "content": """You are an expert AI assistant with advanced reasoning capabilities. Your task is to provide detailed, step-by-step explanations of your thought process. For each step:

1. Provide a clear, concise title.
2. Elaborate on your thought process in the content section.

Response Format:
Use JSON with keys: 'title', 'content'

Key Instructions:
- Acknowledge your limitations as an AI and explicitly state what you can and cannot do.
- Actively explore and evaluate alternative answers or approaches.
- Critically assess your own reasoning; identify potential flaws or biases.
- Incorporate relevant domain knowledge and best practices in your reasoning.
- Consider potential edge cases or exceptions to your reasoning.
- Provide clear justifications for eliminating alternative hypotheses.


Example of a valid JSON response:
```json
{
    "title": "Initial Problem Analysis",
    "content": "To approach this problem effectively, I'll first break down the given information into key components. This involves identifying...[detailed explanation]... By structuring the problem this way, we can systematically address each aspect.",
}```
"""},
    {"role": "assistant",
     "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
]
