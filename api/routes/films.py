from flask import Blueprint, request, jsonify
from marshmallow import ValidationError
from sqlalchemy.exc import IntegrityError

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

@films_router.delete('/<film_id>')
def delete_film(film_id):
    film = Film.query.get(film_id)
    if film is None:
        return {"error": "Film not found"}, 404

    try:
        db.session.delete(film)
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return {
            "error": f"Film {film_id} cannot be deleted because it is referenced in other records."
        }, 409
    
    return film_schema.dump(film), 204

@films_router.put("/<film_id>")
def update_film_full_name(film_id):
    film = Film.query.get(film_id)
    if film is None:
        return {"error": "Film not found"}, 404

    film_data = request.json
    try:
        # validate incoming JSON
        film_schema.load(film_data, partial=False)  # full object required
    except ValidationError as err:
        return jsonify(err.messages), 400

    # overwrite fields
    for key, value in film_data.items():
        if hasattr(film, key):
            setattr(film, key, value)

    db.session.commit()
    return film_schema.dump(film), 200

@films_router.patch("/<film_id>")
def partial_update_film(film_id):
    film = Film.query.get(film_id)
    if film is None:
        return {"error": "Film not found"}, 404

    film_data = request.json
    try:
        # only provided fields validated
        film_schema.load(film_data, partial=True)
    except ValidationError as err:
        return {"error": err.messages}, 400

    # update only given fields
    for key, value in film_data.items():
        if hasattr(film, key):
            setattr(film, key, value)

    db.session.commit()
    return film_schema.dump(film), 200
