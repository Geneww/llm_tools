# -*- coding: utf-8 -*-
"""
@File:        wsgi.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 09, 2024
@Description:
"""
import os
from app import create_app

if not os.path.exists("./logs/"):
    os.mkdir("./logs/")

app = create_app(os.getenv("FLASK_ENV") or "development")


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5001, )
