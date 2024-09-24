# -*- coding: utf-8 -*-
"""
@File:        routes.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 09, 2024
@Description:
"""


def register_routes(api, app, root="api"):
    from app.chat import register_routes as ns_chat
    # 添加路由
    ns_chat(api, app)
