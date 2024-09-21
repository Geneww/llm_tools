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
