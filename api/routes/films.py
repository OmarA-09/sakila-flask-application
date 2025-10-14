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
    train_films = [f for f in films if f.film_id <= 900]
    test_films = [f for f in films if f.film_id > 900]

    train_titles = [f.title for f in train_films]
    train_ratings = [f.rating for f in train_films]
    title_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    rating_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    title_train_enc = title_ohe.fit_transform(np.array(train_titles).reshape(-1, 1))
    rating_train_enc = rating_ohe.fit_transform(np.array(train_ratings).reshape(-1, 1))

    train_features = np.hstack([
        np.array([[f.release_year, f.rental_duration, f.rental_rate, f.length, f.replacement_cost] for f in train_films]),
        title_train_enc,
        rating_train_enc
    ])

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(train_features)
    joblib.dump(kmeans, "api/ml-models/film_kmeans_model.joblib")
    joblib.dump(title_ohe, "api/ml-models/title_ohe.joblib")
    joblib.dump(rating_ohe, "api/ml-models/rating_ohe.joblib")

    # PCA and plot for training data
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(train_features)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap='viridis')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title("Train Film Cluster Visualization")
    plt.colorbar(scatter, label='Cluster')
    img_folder = os.path.join(os.path.dirname(__file__), "plt-images")
    os.makedirs(img_folder, exist_ok=True)
    img_path = os.path.join(img_folder, "film_clusters_train.png")
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

    return {
        "message": "Model trained on films 1-700 and saved.",
        "n_train_films": len(train_films),
        "n_test_films": len(test_films),
        "train_cluster_plot": img_path
    }, 200

# TODO: Ensure that the id passed is outside training data
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

@films_router.get('/evaluate-clusters')
def evaluate_clusters():
    films = Film.query.all()
    test_films = [f for f in films if f.film_id > 900]
    title_ohe = joblib.load("api/ml-models/title_ohe.joblib")
    rating_ohe = joblib.load("api/ml-models/rating_ohe.joblib")
    kmeans = joblib.load("api/ml-models/film_kmeans_model.joblib")

    test_titles = [f.title for f in test_films]
    test_ratings = [f.rating for f in test_films]
    title_test_enc = title_ohe.transform(np.array(test_titles).reshape(-1, 1))
    rating_test_enc = rating_ohe.transform(np.array(test_ratings).reshape(-1, 1))

    test_features = np.hstack([
        np.array([[f.release_year, f.rental_duration, f.rental_rate, f.length, f.replacement_cost] for f in test_films]),
        title_test_enc,
        rating_test_enc
    ])

    test_labels = kmeans.predict(test_features)
    from sklearn.metrics import silhouette_score
    silhouette = silhouette_score(test_features, test_labels) if len(set(test_labels)) > 1 else None

    # PCA and plot for test data
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(test_features)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_2d[:,0], X_2d[:,1], c=test_labels, cmap='viridis')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title("Test Film Cluster Visualization")
    plt.colorbar(scatter, label='Cluster')
    img_folder = os.path.join(os.path.dirname(__file__), "plt-images")
    os.makedirs(img_folder, exist_ok=True)
    img_path = os.path.join(img_folder, "film_clusters_test.png")
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

    return {
        "message": "Model evaluated on test films (>700).",
        "n_test_films": len(test_films),
        "silhouette_score": silhouette,
        "test_cluster_plot": img_path
    }, 200
