# -*- coding: utf-8 -*-
"""
@File:        api.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 09, 2024
@Description:
"""
import json
from flask import Flask, Response
import time
from flask.wrappers import Response
from langchain_community.llms import Ollama

app = Flask(__name__)


# Function to handle streaming responses
def handle_streaming_response(response_stream):
    for chunk in response_stream:
        # Process each chunk as it arrives
        print(chunk, end='')
        yield chunk


def generate_stream():
    llm = Ollama(base_url='http://192.168.124.100:11434', model="llama3-dpo")
    response_stream = llm.stream("为什么天空是蓝色的？", temperature=0.5, top_p=0.5, stream=True)

    return handle_streaming_response(response_stream)


# @app.route()
@app.post('/api/completion/chat')
def stream():
    return Response(generate_stream(), mimetype='text/event-stream')


@app.post('/api/completion/chat/1')
def chat():
    llm = Ollama(base_url='http://192.168.124.100:11434', model="llama3-dpo")
    response = llm.invoke("为什么天空是蓝色的？", temperature=0.5, top_p=0.5)
    print(type(response))

    return Response(json.dumps(response, ensure_ascii=False), mimetype="application/json")


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001, )
