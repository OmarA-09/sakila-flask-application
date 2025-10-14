import math

from flask import Blueprint, request, jsonify, send_file
from marshmallow import ValidationError
from sqlalchemy.exc import IntegrityError

from api.models import db
from api.models.actor import Actor

from api.models.film import Film

from api.schemas.film import film_schema, films_schema
from api.schemas.actor import actors_schema

# Bluerprint gets inserted into flask app
films_router = Blueprint('films', __name__, url_prefix='/films')

def film_to_hateoas(film):
    return {
        **film_schema.dump(film),
        "_links": {
            "self": f"/api/films/{film.film_id}",
            "actors": f"/api/films/{film.film_id}/actors",
            "update": f"/api/films/{film.film_id}",
            "delete": f"/api/films/{film.film_id}",
            "language": f"/api/languages/{film.language_id}"
        }
    }

@films_router.get("")
def read_all_films():
    query = Film.query

    # Filtering
    rating = request.args.get("rating")
    if rating:
        query = query.filter_by(rating=rating)

    title = request.args.get("title")
    if title:
        query = query.filter(Film.title.ilike(f"%{title}%"))

    # Sorting
    sort = request.args.get("sort")
    order = request.args.get("order", "asc")
    if sort in ["rental_rate", "replacement_cost", "release_year"]:
        column = getattr(Film, sort)
        if order == "desc":
            column = column.desc()
        query = query.order_by(column)

    # Pagination
    page = request.args.get("page", type=int, default=1)
    page_size = request.args.get("page_size", type=int, default=20)

    total = query.count()
    films = query.limit(page_size).offset((page - 1) * page_size).all()
    total_pages = math.ceil(total / page_size) if page_size > 0 else 1

    # Dump + enrich each film with language link
    film_items = [film_to_hateoas(f) for f in films]

    base_url = "/api/films"
    links = {
        "self": f"{base_url}?page={page}&page_size={page_size}",
        "first": f"{base_url}?page=1&page_size={page_size}",
        "last": f"{base_url}?page={total_pages}&page_size={page_size}",
        "create": f"{base_url}"
    }
    if page > 1:
        links["prev"] = f"{base_url}?page={page - 1}&page_size={page_size}"
    if page < total_pages:
        links["next"] = f"{base_url}?page={page + 1}&page_size={page_size}"

    return {
        "count": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "items": film_items,
        "_links": links
    }

@films_router.get('/<film_id>')
def read_film(film_id):
    film = Film.query.get(film_id)
    if film is None:
        return {"error": "Film not found"}, 404
    return film_to_hateoas(film)

@films_router.get("/<film_id>/actors")
def read_film_actors(film_id):
    film = Film.query.get(film_id)
    if film is None:
        return {"error": "Film not found"}, 404

    return actors_schema.dump(film.actors), 200

@films_router.post('/')
def create_film():
    film_data = request.json
    actor_ids = film_data.pop("actors", [])

    try:
        film_schema.load(film_data)
    except ValidationError as err:
        return {"error": err.messages}, 400

    film = Film(**film_data)
    db.session.add(film)

    for actor_id in actor_ids:
        actor = Actor.query.get(actor_id)
        if actor:
            film.actors.append(actor)

    db.session.add(film)
    db.session.commit()

    return film_to_hateoas(film), 201

@films_router.delete('/<film_id>')
def delete_film(film_id):
    film = Film.query.get(film_id)
    if film is None:
        return {"error": "Film not found"}, 404

    try:
        db.session.delete(film)
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return {
            "error": f"Film {film_id} cannot be deleted because it is referenced in other records."
        }, 409

    return {
        "message": f"Film {film_id} deleted",
        "_links": {"films": "/api/films"}
    }, 204

def update_film_helper(film_id, film_data, partial):
    film = Film.query.get(film_id)
    if film is None:
        return {"error": "Film not found"}, 404

    actor_ids = film_data.pop("actors", None)

    try:
        film_schema.load(film_data, partial=partial)
    except ValidationError as err:
        return {"error": err.messages}, 400

    # update fields + actors
    for key, value in film_data.items():
        if hasattr(film, key):
            setattr(film, key, value)

    if actor_ids is not None:
        film.actors.clear()
        for actor_id in actor_ids:
            actor = Actor.query.get(actor_id)
            if actor:
                film.actors.append(actor)

    db.session.commit()
    return film_to_hateoas(film), 200

@films_router.put("/<film_id>")
def update_film(film_id):
    return update_film_helper(film_id, request.json, partial=False)

@films_router.patch("/<film_id>")
def partial_update_film(film_id):
    return update_film_helper(film_id, request.json, partial=True)

# ML code
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import io
import random
import os

from sklearn.preprocessing import OneHotEncoder

@films_router.post('/train-clusters')
def train_and_save_clusters():
    films = Film.query.all()

    # Prepare categorical features
    titles = [f.title for f in films]
    ratings = [f.rating for f in films]

    # One-hot encode titles and ratings
    title_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    rating_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    title_encoded = title_ohe.fit_transform(np.array(titles).reshape(-1, 1))
    rating_encoded = rating_ohe.fit_transform(np.array(ratings).reshape(-1, 1))

    # Combine all features
    features = np.hstack([
        np.array([[f.release_year, f.rental_duration, f.rental_rate, f.length, f.replacement_cost] for f in films]),
        title_encoded,
        rating_encoded
    ])

    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(features)
    joblib.dump(kmeans, "api/ml-models/film_kmeans_model.joblib")
    joblib.dump(title_ohe, "api/ml-models/title_ohe.joblib")
    joblib.dump(rating_ohe, "api/ml-models/rating_ohe.joblib")

    # PCA for visualization
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(features)
    pca_loadings = pca.components_
    pc1_importances = np.abs(pca_loadings[0])
    pc2_importances = np.abs(pca_loadings[1])
    pc1_top_idx = np.argmax(pc1_importances)
    pc2_top_idx = np.argmax(pc2_importances)

    feature_names = (
        ["release_year", "rental_duration", "rental_rate", "length", "replacement_cost"] +
        [f"title_{cat}" for cat in title_ohe.categories_[0]] +
        [f"rating_{cat}" for cat in rating_ohe.categories_[0]]
    )
    pc1_feature = feature_names[pc1_top_idx]
    pc2_feature = feature_names[pc2_top_idx]

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap='viridis')
    plt.xlabel(f'PCA 1:  {pc1_feature}')
    plt.ylabel(f'PCA 2:  {pc2_feature}')
    plt.title("Film Cluster Visualization")
    plt.colorbar(scatter, label='Cluster')

    img_folder = os.path.join(os.path.dirname(__file__), "plt-images")
    os.makedirs(img_folder, exist_ok=True)
    img_path = os.path.join(img_folder, "film_clusters.png")
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

    return {
        "message": "Model trained + cluster plot image saved.",
        "n_films": len(films),
        "cluster_plot": img_path,
        "pca_main_features": {
            "PCA1": {
                "feature": pc1_feature,
                "weight": float(pca_loadings[0][pc1_top_idx])
            },
            "PCA2": {
                "feature": pc2_feature,
                "weight": float(pca_loadings[1][pc2_top_idx])
            }
        }
    }, 200

@films_router.get('/predict-cluster/<film_id>')
def predict_film_cluster(film_id):
    film = Film.query.get(film_id)
    if film is None:
        return {"error": "Film not found"}, 404
    kmeans = joblib.load("api/ml-models/film_kmeans_model.joblib")
    title_ohe = joblib.load("api/ml-models/title_ohe.joblib")
    rating_ohe = joblib.load("api/ml-models/rating_ohe.joblib")
    title_encoded = title_ohe.transform(np.array([film.title]).reshape(-1, 1))
    rating_encoded = rating_ohe.transform(np.array([film.rating]).reshape(-1, 1))
    feature = np.hstack([
        np.array([[film.release_year, film.rental_duration, film.rental_rate, film.length, film.replacement_cost]]),
        title_encoded,
        rating_encoded
    ])
    cluster = int(kmeans.predict(feature)[0])
    return jsonify({
        "film_id": film_id,
        "title": film.title,
        "cluster": cluster
    })

@films_router.get('/recommend-like/<film_id>')
def recommend_similar_films(film_id):
    film = Film.query.get(film_id)
    if film is None:
        return {"error": "Film not found"}, 404

    kmeans = joblib.load("api/ml-models/film_kmeans_model.joblib")
    title_ohe = joblib.load("api/ml-models/title_ohe.joblib")
    rating_ohe = joblib.load("api/ml-models/rating_ohe.joblib")
    title_encoded = title_ohe.transform(np.array([film.title]).reshape(-1, 1))
    rating_encoded = rating_ohe.transform(np.array([film.rating]).reshape(-1, 1))
    feature = np.hstack([
        np.array([[film.release_year, film.rental_duration, film.rental_rate, film.length, film.replacement_cost]]),
        title_encoded,
        rating_encoded
    ])
    cluster = int(kmeans.predict(feature)[0])

    films = Film.query.all()
    titles_all = [f.title for f in films]
    ratings_all = [f.rating for f in films]
    title_encoded_all = title_ohe.transform(np.array(titles_all).reshape(-1, 1))
    rating_encoded_all = rating_ohe.transform(np.array(ratings_all).reshape(-1, 1))
    features = np.hstack([
        np.array([[f.release_year, f.rental_duration, f.rental_rate, f.length, f.replacement_cost] for f in films]),
        title_encoded_all,
        rating_encoded_all
    ])
    clusters_all = kmeans.predict(features)

    same_cluster_films = [f for i, f in enumerate(films) if clusters_all[i] == cluster and f.film_id != film.film_id]
    recommends = random.sample(same_cluster_films, min(5, len(same_cluster_films)))

    return {
        "film_id": film_id,
        "title": film.title,
        "cluster": cluster,
        "recommendations": [
            {
                "film_id": f.film_id,
                "title": f.title
            } for f in recommends
        ]
    }, 200
