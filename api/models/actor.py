from api.models import db

class Actor(db.Model):
    actor_id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(45), nullable=False)
    last_name = db.Column(db.String(45), nullable=False)
    last_update = db.Column(
        db.DateTime,
        server_default=db.func.now(), # on POST
        server_onupdate=db.func.now(), # on PUT/PATCH
        nullable=False
    )