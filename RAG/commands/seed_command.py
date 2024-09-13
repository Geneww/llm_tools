# -*- coding: utf-8 -*-
"""
@File:        seed_command.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 10, 2024
@Description:
"""
import numpy as np
import pandas as pd
from datetime import datetime
from flask_script import Command

from app import db
from models.model import AppsManager


def seed_things():
    classes = [AppsManager, ]
    for klass in classes:
        seed_thing(klass)


def seed_thing(cls):
    things = [
        {"name": "Pizza Slicer", "purpose": "Cut delicious pizza"},
        {"name": "Rolling Pin", "purpose": "Roll delicious pizza"},
        {"name": "Pizza Oven", "purpose": "Bake delicious pizza"},
    ]
    db.session.bulk_insert_mappings(cls, things)


class SeedCommand(Command):
    """ Seed the DB."""

    def run(self):
        if (
                input(
                    "Are you sure you want to drop all tables and recreate? (y/N)\n"
                ).lower()
                == "y"
        ):
            print("Dropping tables...")
            db.drop_all()
            db.create_all()
            seed_things()
            db.session.commit()
            print("DB successfully seeded.")
