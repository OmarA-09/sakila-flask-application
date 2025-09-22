# api/models/film_actor.py
from api.models import db

class FilmActor(db.Model):
    film_id = db.Column(db.Integer, db.ForeignKey("film.film_id"), primary_key=True)
    actor_id = db.Column(db.Integer, db.ForeignKey("actor.actor_id"), primary_key=True)
    last_update = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())
