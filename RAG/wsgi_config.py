# -*- coding: utf-8 -*-
"""
@File:        wsgi_config.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 09, 2024
@Description:
"""
# 是否开启debug模式
debug = False
# 访问地址
bind = "0.0.0.0:8000"
# 工作进程数
workers = 4
# 工作线程数
threads = 2
# 超时时间
timeout = 600
# 输出日志级别
loglevel = 'debug'
# 存放日志路径
pidfile = "logs/gunicorn.pid"
# 存放日志路径
accesslog = "logs/access.log"
# 存放日志路径
errorlog = "logs/debug.log"
# gunicorn + apscheduler场景下，解决多worker运行定时任务重复执行的问题
preload_app = True
