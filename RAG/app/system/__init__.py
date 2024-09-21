# -*- coding: utf-8 -*-
"""
@File:        __init__.py.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 13, 2024
@Description:
"""
BASE_ROUTE = "system"


def register_routes(api, app, root="api"):
    from .controllers import api as system_ns
    api.add_namespace(system_ns, path=f"/{root}/{BASE_ROUTE}")
