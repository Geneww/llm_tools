# -*- coding: utf-8 -*-
"""
@File:        manage.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 09, 2024
@Description:
"""
import os
from flask_script import Manager
from app import create_app, db
from commands.seed_command import SeedCommand

env = os.getenv("FLASK_ENV") or "dev"
print(f"Running in {env} environment")
app = create_app(env)

manager = Manager(app)
app.app_context().push()
manager.add_command("seed_db", SeedCommand)


@manager.command
def run():
    app.run()


@manager.command
def init_db():
    print("Creating all tables...")
    db.create_all()


@manager.command
def drop_all():
    if input("确定要清空所有表 (y/N)\n").lower() == "y":
        print("Dropping all tables...")
        db.drop_all()


if __name__ == '__main__':
    manager.run()
