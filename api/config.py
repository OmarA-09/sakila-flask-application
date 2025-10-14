import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass # no .env file

class Config(object):
    TESTING = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False

class ProdConfig(Config):
    SQLALCHEMY_DATABASE_URI = os.getenv('PROD_DATABASE_URI')

class DevConfig(Config):
    SQLALCHEMY_DATABASE_URI = os.getenv('DEV_DATABASE_URI')

match os.getenv('ENV'):
    case 'PRODUCTION':
        config = ProdConfig
    case _:
        config = DevConfig
