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


def generate_secret(length=16):
    import secrets
    # 生成一个16字节长度的随机秘钥
    secret_key = secrets.token_bytes(length)
    return secret_key.hex()


if __name__ == '__main__':
    print(generate_uuid())
    print(generate_secret())
