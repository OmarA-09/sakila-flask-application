from flask import Blueprint
from api.models.language import Language
from api.schemas.language import language_schema, languages_schema

languages_router = Blueprint("languages", __name__, url_prefix="/languages")

@languages_router.get("/")
def read_all_languages():
    return languages_schema.dump(Language.query.all())

@languages_router.get("/<language_id>")
def read_language(language_id):
    language = Language.query.get(language_id)
    if language is None:
        return {"error": "Language not found"}, 404
    return language_schema.dump(language)
