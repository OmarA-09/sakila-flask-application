from flask import Blueprint, request, jsonify
from marshmallow import ValidationError

from api.models import db
from api.models.actor import Actor
from api.schemas.actor import actor_schema, actors_schema

# Bluerprint gets inserted into flask app
actors_router = Blueprint('actors', __name__, url_prefix='/actors')

def actor_to_hateoas(actor):
    return {
        **actor_schema.dump(actor),
        "_links": {
            "self": f"/api/actors/{actor.actor_id}",
            "update": f"/api/actors/{actor.actor_id}",
            "delete": f"/api/actors/{actor.actor_id}",
            "films": f"/api/actors/{actor.actor_id}/films"
        }
    }

@actors_router.get("")
def read_all_actors():
    actors = Actor.query.all()
    actor_items = [actor_to_hateoas(a) for a in actors]

    return {
        "count": len(actor_items),
        "items": actor_items,
        "_links": {
            "self": "/api/actors",
            "create": "/api/actors"
        }
    }

@actors_router.get("/<actor_id>")
def read_actor(actor_id):
    actor = Actor.query.get(actor_id)
    if actor is None:
        return {"error": "Actor not found"}, 404
    return actor_to_hateoas(actor)


@actors_router.post("/")
def create_actor():
    actor_data = request.json
    try:
        actor_schema.load(actor_data)
    except ValidationError as err:
        return jsonify(err.messages), 400

    actor = Actor(**actor_data)
    db.session.add(actor)
    db.session.commit()

    return actor_to_hateoas(actor), 201
@actors_router.delete("/<actor_id>")
def delete_actor(actor_id):
    actor = Actor.query.get(actor_id)
    if actor is None:
        return {"error": "Actor not found"}, 404

    db.session.delete(actor)            #delete record
    db.session.commit()                 #update db

    return {
        "message": f"Actor {actor_id} deleted",
        "_links": {"actors": "/api/actors"}
    }, 204

@actors_router.put("/<actor_id>")
def update_actor_full_name(actor_id):
    actor = Actor.query.get(actor_id)
    if actor is None:
        return {"error": "Actor not found"}, 404

    actor_data = request.json
    try:
        # validate incoming JSON
        actor_schema.load(actor_data, partial=False)  # full object required
    except ValidationError as err:
        return jsonify(err.messages), 400

    # overwrite fields
    actor.first_name = actor_data.get("first_name")
    actor.last_name = actor_data.get("last_name")

    db.session.commit()
    return actor_to_hateoas(actor), 200

@actors_router.patch("/<actor_id>")
def partial_update_actor(actor_id):
    actor = Actor.query.get(actor_id) # retrieve actor object
    if actor is None:
        return {"error": "Actor not found"}, 404

    actor_data = request.json # python dict
    try:
        actor_schema.load(actor_data, partial=True) # validate dict against schema
    except ValidationError as err:
        return jsonify(err.messages), 400 # jsonify python object

    if actor_data.get("first_name"):
        actor.first_name = actor_data.get("first_name") #update object
    elif actor_data.get("last_name"):
        actor.last_name = actor_data.get("last_name")

    db.session.commit()
    return actor_to_hateoas(actor), 200
