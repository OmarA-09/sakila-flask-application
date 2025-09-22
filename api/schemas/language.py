from api.schemas import ma
from api.models.language import Language
from marshmallow import fields

class LanguageSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Language

    language_id = fields.Int(dump_only=True)
    last_update = fields.DateTime(dump_only=True)

language_schema = LanguageSchema()
languages_schema = LanguageSchema(many=True)
