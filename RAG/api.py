# -*- coding: utf-8 -*-
"""
@File:        api.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 09, 2024
@Description:
"""
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


@app.route('/chat')
def stream():
    return Response(generate_stream(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True)
