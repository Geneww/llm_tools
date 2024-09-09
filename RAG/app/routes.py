# -*- coding: utf-8 -*-
"""
@File:        toutes.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 09, 2024
@Description:
"""


def register_routes(api, app, root="api"):
    from app.chat import register_routes as chat
    # 添加路由
    chat(api, app)
