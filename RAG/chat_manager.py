# -*- coding: utf-8 -*-
"""
@File:        chat_manager.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 07, 2024
@Description:
    RAG:
    Query -> 检索 -> rerank-> Prompt -> LLM -> response

    检索：
        加载文档
        文档切分
        embedding
        入库

"""
import re
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma

from config import *

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)


def build_prompt(prompt_template, **kwargs):
    """
    构建prompt模板
    :param prompt_template:
    :param kwargs:
    :return:
    """
    prompt = prompt_template
    for k, v in kwargs.items():
        if isinstance(v, str):
            val = v
        elif isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n'.join(v)
        else:
            val = str(v)
        prompt = prompt.replace(f'__{k.upper()}__', val)
    return prompt


def contains_english_letter(s):
    return bool(re.search('[a-zA-Z]', s))


def extract_text_from_pdf(filename, page_numbers=None, min_line_length=1):
    """
    从PDF文件中(按照指定页码)提取文字
    :param filename:
    :param page_numbers:
    :param min_line_length:
    :return:
    """
    paragraphs = []
    buffer = ''
    full_text = ''
    # 提取全部文本信息
    for i, page_layout in enumerate(extract_pages(filename)):
        # 如果指定了页码范围，跳过范围外的页面
        if page_numbers is not None and i not in page_numbers:
            continue
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'
    # 按空行分隔，将文本重新组织成段落
    lines = full_text.split("\n")
    for text in lines:
        if len(text) >= min_line_length:
            # 如果不含英文去除中文间空格
            if not contains_english_letter(text):
                text = text.strip(' ')
            buffer += (" " + text) if not text.endswith('-') else text.strip('-')
        elif buffer:
            paragraphs.append(buffer)
            buffer = ""
        else:
            pass
    if buffer:
        paragraphs.append(buffer)
    return paragraphs


def test_rag():
    user_query = "how many parameters does llama 2 have?"
    # 1.检索
    # 2.构建prompt
    prompt = build_prompt(PROMPT_TEMPLATE, info=["llama2", "llama3"], query=user_query)
    print("========Prompt=========")
    print(prompt)
    # 3.调用LLM


if __name__ == '__main__':
    a = extract_text_from_pdf("/Users/gene/Downloads/MakerWorld中国站-创作者答疑.pdf")
    print(a)
    # test_rag()
    docs = text_splitter.split_text(a)
    print(docs)