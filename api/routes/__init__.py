from flask import Blueprint

from api.routes.actors import actors_router
from api.routes.films import films_router

routes = Blueprint('api', __name__, url_prefix='/api')

routes.register_blueprint(actors_router)
routes.register_blueprint(films_router)