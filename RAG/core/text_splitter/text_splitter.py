# -*- coding: utf-8 -*-
"""
@File:        text_splitter.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 19, 2024
@Description:
"""
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextSplitter:
    @classmethod
    def recursive_split_text(cls, text: str) -> List[str]:
        """
        这个参数定义了相邻文本块之间的重叠字符数。重叠部分有助于保持文本的连贯性，尤其是在处理需要上下文理解的任务时。
        一个常见的做法是将 chunk_overlap 设置为 chunk_size 的10-20%，这样可以在保持连贯性的同时避免过多的重复
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "。", ".", " ", ""]
        )
        return text_splitter.split_text(text)


if __name__ == '__main__':
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader("/Users/gene/Library/Mobile Documents/com~apple~CloudDocs/Downloads/2024达亿瓦综合目录书.pdf")
    pages = loader.load_and_split()
    print(pages)
    docs = TextSplitter.recursive_split_text(pages)
    print(len(docs))
