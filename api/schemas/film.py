from api.models.film import Film
from api.schemas import ma
from marshmallow import fields

class FilmSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Film

    film_id = fields.Int(dump_only=True)
    last_update = fields.DateTime(dump_only=True)

# instantiate
film_schema = FilmSchema()
films_schema = FilmSchema(many=True)