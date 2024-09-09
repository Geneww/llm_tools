# -*- coding: utf-8 -*-
"""
@File:        tools.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 09, 2024
@Description:
"""
import uuid


def generate_uuid():
    return uuid.uuid4().hex


if __name__ == '__main__':
    print(generate_uuid())
