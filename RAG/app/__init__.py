# -*- coding: utf-8 -*-
"""
@File:        __init__.py.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 09, 2024
@Description:
"""

from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import DateTime
from flask_restx import Api

from app.app_global import AppGlobal
from app.logger import Loggers
from flask_jwt_extended import JWTManager
from common.tools import datetime_to_string
from flask_apscheduler import APScheduler

db = SQLAlchemy()
app_global = AppGlobal()
logger = Loggers()
jwt = JWTManager()
scheduler = APScheduler()


def to_dict(self, all=False):
    res = {}
    for c in self.__table__.columns:
        if c.name == "deleted_at" and not all:
            continue

        res[c.name] = getattr(self, c.name)
        if isinstance(c.type, DateTime):
            v = getattr(self, c.name)
            if v:
                res[c.name] = datetime_to_string(v)

    return res


db.Model.to_dict = to_dict


def create_app(env=None):
    from app.config import config_by_name
    from app.routes import register_routes
    from core.llm.llm_builder import LLMBuilder
    from core.vector_store.faiss_vector_store_client import FaissVectorStore
    from models.model import AppsManager, User

    app = Flask(__name__)
    app.config.from_object(config_by_name[env or "dev"])
    api = Api(app, version="1.0", title="itqm-aigc", description="""itqm生成式AI服务
```json
全局返回体:
{
    "code": code,  // 状态码 int 200为正常, 40x为异常
    "message": message, // 响应消息提示 string
    "data": data,  // 响应数据 object
    "timestamp": int(time.time())  // 秒级时间戳
}
```
---
sign 签名方式: md5(app_id + nonce + timestamp + secret); nonce为 4位数字字符串(1000 - 9999)

    """, doc="/doc")

    register_routes(api, app)
    db.init_app(app)
    jwt.init_app(app)
    scheduler.init_app(app)
    app_global.init(config_by_name[env or "dev"], app)

    with app.app_context():
        db.create_all()
        AppsManager.init_record()  # 初始化默认应用
        User.init_admin()  # 初始化管理员账号

    # 初始化向量数据
    embedding = LLMBuilder.to_llm_embedding()
    faiss_vector = FaissVectorStore(embedding)
    # faiss_vector.try_reload_db(app)
    # threading.Thread(
    #     target=faiss_vector.auto_refresh_task,
    #     kwargs={
    #         "flask_app": app
    #     }
    # ).start()

    app_global.load_faiss_vector_index(faiss_vector)

    scheduler.start()

    @app.route("/health")
    def health():
        return jsonify("healthy")

    return app
