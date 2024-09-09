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
from flask_re

app = Flask(__name__)


def generate_stream():
    for i in range(10):
        yield f"data: {i}\n\n"
        time.sleep(1)  # 模拟数据处理的延迟


@app.route('/chat')
@app.doc()
def stream():
    return Response(generate_stream(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True)
