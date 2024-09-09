# -*- coding: utf-8 -*-
"""
@File:        t4.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 07, 2024
@Description:
"""
import requests


def stream_response(url):
    with requests.get(url, stream=True) as response:
        try:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if line:  # filter out keep-alive new lines
                    print(line)
        except requests.exceptions.HTTPError as e:
            print(e)


if __name__ == '__main__':
    port = 8000
    base_url = "http://192.168.124.100:{}/v1".format(port)
    stream_url = "{}/stream".format(base_url)
    stream_response(stream_url)
