from flask import Blueprint, request, jsonify
from marshmallow import ValidationError
from sqlalchemy.exc import IntegrityError

from api.models import db
from api.models.actor import Actor
from api.models.film_actor import FilmActor
from api.models.film import Film

from api.schemas.film import film_schema, films_schema
from api.schemas.actor import actors_schema

# Bluerprint gets inserted into flask app
films_router = Blueprint('films', __name__, url_prefix='/films')

@films_router.get('/')
def read_all_films():
    query = Film.query

    # Filtering
    rating = request.args.get("rating")
    if rating:
        query = query.filter_by(rating=rating)

    title = request.args.get("title")
    if title:
        query = query.filter(Film.title.ilike(f"%{title}%"))

    # Sorting
    sort = request.args.get("sort")
    order = request.args.get("order", "asc")
    if sort in ["rental_rate", "replacement_cost", "release_year"]:
        column = getattr(Film, sort)
        if order == "desc":
            column = column.desc()
        query = query.order_by(column)

    # Pagination
    limit = request.args.get("limit", type=int, default=20)
    offset = request.args.get("offset", type=int, default=0)
    films = query.limit(limit).offset(offset).all()

    return films_schema.dump(films)


@films_router.get('/<film_id>')
def read_film(film_id):
    film = Film.query.get(film_id)
    if film is None:
        return {"error": "Film not found"}, 404
    return film_schema.dump(film)

@films_router.get("/<film_id>/actors")
def read_film_actors(film_id):
    # check film exists first
    film = Film.query.get(film_id)
    if film is None:
        return {"error": "Film not found"}, 404

    # find all actor_ids linked in film_actor
    film_actors = FilmActor.query.filter_by(film_id=film_id).all()
    actor_ids = [film_actor.actor_id for film_actor in film_actors]

    if not actor_ids:
        return {"actors": []}, 200

    # fetch actors
    actors = Actor.query.filter(Actor.actor_id.in_(actor_ids)).all()
    return actors_schema.dump(actors), 200

@films_router.post('/')
def create_film():
    film_data = request.json
    actor_ids = film_data.pop("actors", [])

    try:
        film_schema.load(film_data)
    except ValidationError as err:
        return {"error": err.messages}, 400

    film = Film(**film_data)
    db.session.add(film)

    db.session.flush()  # assign film_id before commit so we can get film.film_id for filmActor

    # manually link actors via FilmActor rows
    for actor_id in actor_ids:
        actor = Actor.query.get(actor_id)
        if actor:
            db.session.add(
                FilmActor(
                    film_id=film.film_id, actor_id=actor.actor_id
                )
            )

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
