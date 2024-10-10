# -*- coding: utf-8 -*-
"""
@File:        test_server.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 26, 2024
@Description:
"""
import requests


def stream_events(url, data=None, headers=None):
    # 发送POST请求
    with requests.post(url, data=data, headers=headers, stream=True) as response:
        try:
            # 确保响应状态码为200
            response.raise_for_status()
            # 循环读取流式响应的内容
            for line in response.iter_lines():
                if line:
                    # 解析每行数据
                    line = line.decode('utf-8')
                    print(line)
        except requests.exceptions.RequestException as e:
            # 处理请求异常
            print(e)


if __name__ == "__main__":
    import json
    # 替换为你的SSE服务URL
    url = 'http://192.168.124.20:5001/api/completion/chat'
    # 可选：POST请求的数据
    data = {
        "app_id": "1001",
        "nonce": "1000",
        "sign": "1dc31e8d26b003a03def09e074682b5b",
        "timestamp": 1727101560,
        "conversation_id": "12345",
        "query": "为什么天空是蓝色的?",
        "conversation_type": "common",
        "temperature": 1.0,
        "response_mode": "stream"
    }
    # 可选：自定义请求头
    headers = {'Authorization': 'Bearer your_token_here'}
    # stream_events(url, data=json.dumps(data))

    import queue
    import threading

    msg_queue = queue.Queue()


