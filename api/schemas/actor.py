from api.models.actor import Actor
from api.schemas import ma

class ActorSchema(ma.ModelSchema):
    class Meta:
        model = Actor

# instantiate
actor_schema = ActorSchema()
actors_schema = ActorSchema(many=True)