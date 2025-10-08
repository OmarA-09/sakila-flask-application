import os
from dotenv import load_dotenv

load_dotenv()

class Config(object):
    TESTING = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False

class ProdConfig(Config):
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URI')

class DevConfig(Config):
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URI')

match os.getenv('ENV'):
    case 'PRODUCTION':
        config = ProdConfig
    case _:
        config = DevConfig
