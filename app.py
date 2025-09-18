from flask import Flask

from api.config import config
from api.routes import routes

def create_app():
    app = Flask(__name__)
    app.config.from_object(config)

    from api.models import db
    db.init_app(app)

    from api.schemas import ma
    ma.init_app(app)

    app.register_blueprint(routes)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run()
