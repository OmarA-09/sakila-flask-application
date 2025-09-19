from flask import Blueprint, request, jsonify
from marshmallow import ValidationError

from api.models import db
from api.models.film import Film
from api.schemas.film import film_schema, films_schema

# Bluerprint gets inserted into flask app
films_router = Blueprint('films', __name__, url_prefix='/films')

@films_router.get('/')
def read_all_films():
    films = Film.query.all()
    return films_schema.dump(films)

@films_router.get('/<film_id>')
def read_film(film_id):
    film = Film.query.get(film_id)
    if film is None:
        return {"error": "Film not found"}, 404
    return film_schema.dump(film)

@films_router.post('/')
def create_film():
    film_data = request.json

    try:
        film_schema.load(film_data)
    except ValidationError as err:
        return {"error": err.messages}, 400

    film = Film(**film_data)
    db.session.add(film)
    db.session.commit()

    return film_schema.dump(film), 201
'''


    actor = Actor(**actor_data)         # Create a new actor model
    db.session.add(actor)               # Insert the record
    db.session.commit()                 # Update the database

    return actor_schema.dump(actor), 201     # Serialize the created actor

'''