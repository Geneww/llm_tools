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

from app.logger import Loggers
from flask_jwt_extended import JWTManager
from flask_apscheduler import APScheduler

db = SQLAlchemy()
logger = Loggers()
jwt = JWTManager()
scheduler = APScheduler()


def create_app(env=None):
    from app.config import config_by_name
    from app.routes import register_routes
    # from core.llm.llm_builder import LLMBuilder

    app = Flask(__name__)
    app.config.from_object(config_by_name[env or "dev"])
    api = Api(app, version="1.0", title="LLM api", description="""
LLM chat服务
---
```json
全局返回体:
{
    "code": code,                  // 状态码 int 0为正常, 4x为异常
    "message": message,            // 响应消息提示 string
    "data": data,                  // 响应数据 object
    "timestamp": int(time.time())  // 秒级时间戳
}
```
---
```json
状态码:
   "0"     :  "请求成功",
   "4001"  :  "数据库查询错误",
   "4002"  :  "无数据",
   "4003"  :  "数据已存在",
   "4004"  :  "数据错误",
   "4101"  :  "用户未登录",
   "4102"  :  "用户登录失败",
   "4103"  :  "参数错误",
   "4104"  :  "用户不存在或未激活",
   "4105"  :  "用户身份错误",
   "4106"  :  "密码错误",
   "4201"  :  "非法请求或请求次数受限",
   "4202"  :  "IP受限",
   "4301"  :  "第三方系统错误",
   "4302"  :  "文件读写错误",
   "4500"  :  "内部错误",
   "4501"  :  "未知错误",          
```
---
```text
sign 签名方式: md5(app_id + nonce + timestamp + secret); nonce为 4位数字字符串(1000 - 9999)
```
""", doc="/doc")

    register_routes(api, app)
    db.init_app(app)
    jwt.init_app(app)
    scheduler.init_app(app)

    with app.app_context():
        db.create_all()
    #     AppsManager.init_record()  # 初始化默认应用
    #     User.init_admin()  # 初始化管理员账号
    #
    # # 初始化向量数据
    # embedding = LLMBuilder.to_llm_embedding()
    # faiss_vector = FaissVectorStore(embedding)
    # faiss_vector.try_reload_db(app)
    # threading.Thread(
    #     target=faiss_vector.auto_refresh_task,
    #     kwargs={
    #         "flask_app": app
    #     }
    # ).start()

    scheduler.start()

    @app.route("/health")
    def health():
        return jsonify("healthy")

    return app
