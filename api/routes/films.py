import math

from flask import Blueprint, request, jsonify
from marshmallow import ValidationError
from sqlalchemy.exc import IntegrityError

from api.models import db
from api.models.actor import Actor

from api.models.film import Film

from api.schemas.film import film_schema, films_schema
from api.schemas.actor import actors_schema

# Bluerprint gets inserted into flask app
films_router = Blueprint('films', __name__, url_prefix='/films')

@films_router.get("")
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
    page = request.args.get("page", type=int, default=1)
    page_size = request.args.get("page_size", type=int, default=20)

    total = query.count()
    films = query.limit(page_size).offset((page - 1) * page_size).all()
    total_pages = math.ceil(total / page_size) if page_size > 0 else 1

    # Dump + enrich each film with language link
    film_items = []
    for film in films:
        film_data = film_schema.dump(film)
        film_data["_links"] = {
            "self": f"/api/films/{film.film_id}",
            "language": f"/api/languages/{film.language_id}"
        }
        film_items.append(film_data)

    # Hypermedia controls
    base_url = "/api/films"
    links = {
        "self": f"{base_url}?page={page}&page_size={page_size}",
        "first": f"{base_url}?page=1&page_size={page_size}",
        "last": f"{base_url}?page={total_pages}&page_size={page_size}"
    }
    if page > 1:
        links["prev"] = f"{base_url}?page={page-1}&page_size={page_size}"
    if page < total_pages:
        links["next"] = f"{base_url}?page={page+1}&page_size={page_size}"

    return {
        "count": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "items": film_items,
        "_links": links
    }

@films_router.get('/<film_id>')
def read_film(film_id):
    film = Film.query.get(film_id)
    if film is None:
        return {"error": "Film not found"}, 404

    film_data = film_schema.dump(film)

    # Add hypermedia link for language
    film_data["_links"] = {
        "self": f"/api/films/{film_id}",
        "language": f"/api/languages/{film.language_id}"
    }

    return film_data

@films_router.get("/<film_id>/actors")
def read_film_actors(film_id):
    film = Film.query.get(film_id)
    if film is None:
        return {"error": "Film not found"}, 404

    return actors_schema.dump(film.actors), 200

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

    for actor_id in actor_ids:
        actor = Actor.query.get(actor_id)
        if actor:
            film.actors.append(actor)

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

def update_film_helper(film_id, film_data, partial):
    film = Film.query.get(film_id)
    if film is None:
        return {"error": "Film not found"}, 404

    actor_ids = film_data.pop("actors", None)

    try:
        film_schema.load(film_data, partial=partial)
    except ValidationError as err:
        return {"error": err.messages}, 400

    # update fields + actors
    for key, value in film_data.items():
        if hasattr(film, key):
            setattr(film, key, value)

    if actor_ids is not None:
        film.actors.clear()
        for actor_id in actor_ids:
            actor = Actor.query.get(actor_id)
            if actor:
                film.actors.append(actor)

    db.session.commit()
    return film_schema.dump(film), 200

@films_router.put("/<film_id>")
def update_film(film_id):
    return update_film_helper(film_id, request.json, partial=False)

@films_router.patch("/<film_id>")
def partial_update_film(film_id):
    return update_film_helper(film_id, request.json, partial=True)

