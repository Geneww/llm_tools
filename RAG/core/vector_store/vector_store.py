# -*- coding: utf-8 -*-
"""
@File:        vector_store.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 19, 2024
@Description:
"""
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document


class FAISSVectorStore:
    pass

    def __init__(self):
        pass

    def loder_dir(self):
        """加载目录内文件"""
