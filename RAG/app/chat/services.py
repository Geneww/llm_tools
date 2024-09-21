# -*- coding: utf-8 -*-
"""
@File:        services.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 09, 2024
@Description:
"""
from langchain_community.llms import Ollama


class ChatServices:
    @classmethod
    def generator(cls):
        llm = Ollama(base_url='http://192.168.124.100:11434', model="llama3-cot")
        response = llm.invoke("hello!", temperature=0.7, top_p=0.9)

        prompt = generate_prompt(user_query)
        # sa = StreamableOpenAI()
        # print(sa.generate(prompts=[user_query]))
        response = llm.invoke(user_query)
        print(response)


def test(name):
    llm = Ollama(base_url='http://192.168.124.100:11434', model="llama3-dpo")
    response = llm.invoke("为什么天空是蓝色的？", temperature=0.5, top_p=0.5)
    print(f"id:{name}- {response}")


if __name__ == '__main__':
    import threading
    for i in range(5):
        t = threading.Thread(target=test, kwargs={
            "name": i
        })
        t.start()
        print(t.name)
    print("done")
