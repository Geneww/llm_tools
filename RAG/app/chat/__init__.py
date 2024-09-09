# -*- coding: utf-8 -*-
"""
@File:        __init__.py.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 09, 2024
@Description:
"""
BASE_ROUTE = "completion"


def register_routes(api, app, root="api"):
    from .controllers import api as chat_ns
    api.add_namespace(chat_ns, path=f"/{root}/{BASE_ROUTE}")

