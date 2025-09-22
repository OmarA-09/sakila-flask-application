from api.models import db, film_actor

class Film(db.Model):
    film_id = db.Column(db.SmallInteger, primary_key=True)
    title = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)
    release_year = db.Column(db.Integer)  # YEAR -> int
    language_id = db.Column(db.SmallInteger, nullable=False)
    # ignoring original_language_id
    rental_duration = db.Column(db.SmallInteger, nullable=False, default=3)
    rental_rate = db.Column(db.Numeric(4, 2), nullable=False, default=4.99) # Numeric(num_digits, decimal_places)
    length = db.Column(db.SmallInteger)
    replacement_cost = db.Column(db.Numeric(5, 2), nullable=False, default=19.99)

    rating = db.Column(
        db.Enum("G", "PG", "PG-13", "R", "NC-17", name="rating_enum"),
        nullable=False
    )

    # treat as string CSV for MVP - handle SET later
    special_features = db.Column(db.String(255))

    last_update = db.Column(
        db.DateTime,
        server_default=db.func.now(),
        server_onupdate=db.func.now(),
        nullable=False
    )

    actors = db.relationship("Actor", secondary=film_actor, back_populates="films")


