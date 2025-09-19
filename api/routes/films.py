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
'''

@actors_router.post('/')
def create_actor():
    actor_data = request.json   # Get the parsed request body

    try:
        actor_schema.load(actor_data)   # Validate against the schema
    except ValidationError as err:
        return jsonify(err.messages), 400

    actor = Actor(**actor_data)         # Create a new actor model
    db.session.add(actor)               # Insert the record
    db.session.commit()                 # Update the database

    return actor_schema.dump(actor), 201     # Serialize the created actor

'''