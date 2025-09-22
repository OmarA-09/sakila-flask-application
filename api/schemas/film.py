from api.models.film import Film
from api.schemas import ma
from api.schemas.actor import ActorSchema
from marshmallow import fields, ValidationError

ALLOWED_FEATURES = {"Trailers", "Commentaries", "Deleted Scenes", "Behind the Scenes"}

# ensures price format
def validate_price(value):
    try:
        s = str(value)
    except Exception:
        raise ValidationError("Must be a number.")
    if "." in s:
        decimals = s.split(".")[1]
        if len(decimals) > 2:
            raise ValidationError("Must have at most 2 decimal places.")

# ensures csv format
def validate_special_features_csv(value):
    if not value:
        return
    # require exact matches, no leading/trailing spaces allowed
    features = value.split(",")
    invalid = [f for f in features if f not in ALLOWED_FEATURES]
    if invalid:
        raise ValidationError(
            f"Invalid features: {', '.join(invalid)}. "
            f"Allowed: {', '.join(sorted(ALLOWED_FEATURES))}"
            f"Make sure there are no spaces after commas."
        )

class FilmSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Film
        include_relationships = True

    film_id = fields.Int(dump_only=True)
    last_update = fields.DateTime(dump_only=True)

    rental_rate = fields.Float(required=True, validate=validate_price)
    replacement_cost = fields.Float(required=True, validate=validate_price)

    # CSV string that must only contain allowed options
    special_features = fields.String(validate=validate_special_features_csv)

    actors = fields.Nested(ActorSchema, many=True, dump_only=True)


film_schema = FilmSchema()
films_schema = FilmSchema(many=True)
