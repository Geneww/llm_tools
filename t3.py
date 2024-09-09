# -*- coding: utf-8 -*-
"""
@File:        t3.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 07, 2024
@Description:
"""
import re


def contains_english_letter(s):
    return bool(re.search('[a-zA-Z]', s))


# Example usage
string1 = "1234"
string2 = "Hello123"
string3 = "你好"

print(contains_english_letter(string1))  # Output: False
print(contains_english_letter(string2))  # Output: True
print(contains_english_letter(string3))  # Output: False
import requests

url = "http://192.168.124.100:8000/v1/chat/completions"
# url = "http://192.168.124.100:8000/v1/models"
data = {
    "body": "123"
}
res = requests.post(url=url, data=data)
print(res.text)

import os
from openai import OpenAI
import openai


def main():
    # Change to your custom port
    port = 8000
    client = OpenAI(
        api_key="your_api_key_here",  # replace with your actual API key
        base_url="http://192.168.124.100:{}/v1".format(os.environ.get("API_PORT", port)),
    )

    messages = []
    messages.append({"role": "user", "content": "hello, where is USA"})
    # Create a completion with streaming
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Adjust model as needed
        messages=messages,
        temperature=0.8,
        top_p=0.7,
        stream=True
    )

    # Iterate over the stream of events
    for event in response:
        # delta = event["choices"][0]["delta"]
        delta = event.choices[0].delta
        if delta.content is not None:
            stream = delta.content
            print(stream)
        else:
            print(delta)


if __name__ == '__main__':
    import time
    # main()
    timestamp = str(int(time.time()))
    print(timestamp)
    from datetime import datetime, timedelta, timezone

    # UNIX 时间戳
    timestamp = 1725867365

    # 将时间戳转换为 UTC 时间
    utc_time = datetime.utcfromtimestamp(timestamp)

    # 转换为北京时间 (UTC+8)
    beijing_time = utc_time + timedelta(hours=8)

    # 打印北京时间
    print(beijing_time.strftime('%Y-%m-%d %H:%M:%S'))
