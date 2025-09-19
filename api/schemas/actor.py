from api.models.actor import Actor
from api.schemas import ma
from marshmallow import fields

class ActorSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Actor

    actor_id = fields.Int(dump_only=True)
    last_update = fields.DateTime(dump_only=True)

# instantiate
actor_schema = ActorSchema()
actors_schema = ActorSchema(many=True)