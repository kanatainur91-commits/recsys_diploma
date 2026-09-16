from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# APPLICATION
# =========================================================

app = FastAPI(
    title="Интеллектуалды ұсыныстар жүйесі",
    description="Thompson Sampling негізіндегі фильмдер ұсынысы",
    version="1.0"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

PUBLIC_DATA = BASE_DIR / "public_data"
TEMPLATES = BASE_DIR / "templates"


# =========================================================
# LOAD DATA
# =========================================================

movies_path = PUBLIC_DATA / "movies.csv"
bandit_path = PUBLIC_DATA / "bandit_stats.csv"
history_path = PUBLIC_DATA / "user_history.csv"


movies = pd.read_csv(movies_path)
bandit_stats = pd.read_csv(bandit_path)
user_history_df = pd.read_csv(history_path)


# =========================================================
# PREPARE MOVIE DATA
# =========================================================

# Проверяем название ID фильма
if "movie_id" not in movies.columns:

    if "movieId" in movies.columns:
        movies = movies.rename(
            columns={"movieId": "movie_id"}
        )

    elif "id" in movies.columns:
        movies = movies.rename(
            columns={"id": "movie_id"}
        )


# =========================================================
# PREPARE BANDIT DATA
# =========================================================

if "movie_id" not in bandit_stats.columns:

    if "movieId" in bandit_stats.columns:
        bandit_stats = bandit_stats.rename(
            columns={"movieId": "movie_id"}
        )

    elif "id" in bandit_stats.columns:
        bandit_stats = bandit_stats.rename(
            columns={"id": "movie_id"}
        )


# Если alpha/beta отсутствуют, создаём их
if "alpha" not in bandit_stats.columns:
    bandit_stats["alpha"] = 1.0

if "beta" not in bandit_stats.columns:
    bandit_stats["beta"] = 1.0


# =========================================================
# USER HISTORY
# =========================================================

user_history = {}

if not user_history_df.empty:

    # Возможные названия столбцов
    if "user_id" not in user_history_df.columns:

        if "userId" in user_history_df.columns:
            user_history_df = user_history_df.rename(
                columns={"userId": "user_id"}
            )

    if "movie_id" not in user_history_df.columns:

        if "movieId" in user_history_df.columns:
            user_history_df = user_history_df.rename(
                columns={"movieId": "movie_id"}
            )


    if (
        "user_id" in user_history_df.columns
        and "movie_id" in user_history_df.columns
    ):

        for _, row in user_history_df.iterrows():

            uid = int(row["user_id"])
            mid = int(row["movie_id"])

            if uid not in user_history:
                user_history[uid] = set()

            user_history[uid].add(mid)


# =========================================================
# USER PREFERENCES
# =========================================================

user_liked_genres = {}


# =========================================================
# FEEDBACK MODEL
# =========================================================

class Feedback(BaseModel):

    user_id: int
    item_id: int
    reward: int


# =========================================================
# MAIN PAGE
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home():

    index_path = TEMPLATES / "index.html"

    if not index_path.exists():

        return """
        <h1>Қате</h1>
        <p>index.html файлы табылмады.</p>
        """

    return index_path.read_text(
        encoding="utf-8"
    )


# =========================================================
# RECOMMEND
# =========================================================

@app.get("/recommend")
def recommend(
    user_id: int,
    k: int = 10
):

    if k < 1:
        k = 10

    # Уже просмотренные фильмы
    watched = user_history.get(
        user_id,
        set()
    )

    # Копия данных
    candidates = movies.copy()

    # Исключаем просмотренные
    if watched:

        candidates = candidates[
            ~candidates["movie_id"].isin(watched)
        ]


    # Объединяем с alpha/beta
    candidates = candidates.merge(
        bandit_stats[
            [
                "movie_id",
                "alpha",
                "beta"
            ]
        ],
        on="movie_id",
        how="left"
    )


    # Если alpha/beta отсутствуют
    candidates["alpha"] = (
        candidates["alpha"]
        .fillna(1.0)
    )

    candidates["beta"] = (
        candidates["beta"]
        .fillna(1.0)
    )


    # =====================================================
    # THOMPSON SAMPLING
    # =====================================================

    candidates["score"] = candidates.apply(
        lambda row: np.random.beta(
            max(float(row["alpha"]), 0.01),
            max(float(row["beta"]), 0.01)
        ),
        axis=1
    )


    # Сортировка
    candidates = candidates.sort_values(
        "score",
        ascending=False
    )


    result = candidates.head(k)


    # Формируем ответ
    response = []

    for _, row in result.iterrows():

        response.append({

            "movie_id": int(
                row["movie_id"]
            ),

            "title": str(
                row["title"]
            ),

            "genres": str(
                row.get(
                    "genres",
                    ""
                )
            ),

            "score": float(
                row["score"]
            )

        })


    return response


# =========================================================
# FEEDBACK
# =========================================================

@app.post("/feedback")
def feedback(data: Feedback):

    user_id = int(data.user_id)
    item_id = int(data.item_id)
    reward = int(data.reward)


    # Проверяем reward
    if reward not in [0, 1]:

        raise HTTPException(
            status_code=400,
            detail="Reward must be 0 or 1"
        )


    # Проверяем фильм
    movie_exists = (
        movies["movie_id"] == item_id
    ).any()


    if not movie_exists:

        raise HTTPException(
            status_code=404,
            detail="Фильм табылмады"
        )


    # =====================================================
    # FIND BANDIT STATE
    # =====================================================

    row_index = bandit_stats.index[
        bandit_stats["movie_id"] == item_id
    ]


    # Если фильма нет в bandit_stats
    if len(row_index) == 0:

        new_row = {
            "movie_id": item_id,
            "alpha": 1.0,
            "beta": 1.0
        }

        bandit_stats.loc[
            len(bandit_stats)
        ] = new_row

        row_index = bandit_stats.index[
            bandit_stats["movie_id"] == item_id
        ]


    idx = row_index[0]


    # =====================================================
    # THOMPSON PARAMETERS UPDATE
    # =====================================================

    old_alpha = float(
        bandit_stats.at[idx, "alpha"]
    )

    old_beta = float(
        bandit_stats.at[idx, "beta"]
    )


    if reward == 1:

        new_alpha = old_alpha + 1

        new_beta = old_beta

    else:

        new_alpha = old_alpha

        new_beta = old_beta + 1


    bandit_stats.at[
        idx,
        "alpha"
    ] = new_alpha


    bandit_stats.at[
        idx,
        "beta"
    ] = new_beta


    # =====================================================
    # USER HISTORY
    # =====================================================

    if user_id not in user_history:

        user_history[user_id] = set()


    user_history[user_id].add(
        item_id
    )


    # =====================================================
    # LIKED GENRES
    # =====================================================

    if reward == 1:

        movie_row = movies[
            movies["movie_id"] == item_id
        ].iloc[0]


        genres = str(
            movie_row.get(
                "genres",
                ""
            )
        )


        if user_id not in user_liked_genres:

            user_liked_genres[user_id] = []


        for genre in genres.split("|"):

            if (
                genre
                and genre != "(no genres listed)"
                and genre not in user_liked_genres[user_id]
            ):

                user_liked_genres[user_id].append(
                    genre
                )


    # =====================================================
    # RESPONSE
    # =====================================================

    return {

        "status": "success",

        "message": "Баға сәтті қабылданды",

        "user_id": user_id,

        "item_id": item_id,

        "reward": reward,

        "alpha": new_alpha,

        "beta": new_beta

    }


# =========================================================
# FINAL RECOMMENDATIONS
# =========================================================

@app.get("/final-recommend")
def final_recommend(
    user_id: int,
    k: int = 5
):

    if k < 1:
        k = 5


    # Просмотренные фильмы
    watched = user_history.get(
        user_id,
        set()
    )


    candidates = movies.copy()


    # Исключаем просмотренные
    if watched:

        candidates = candidates[
            ~candidates["movie_id"].isin(watched)
        ]


    # =====================================================
    # BANDIT DATA
    # =====================================================

    candidates = candidates.merge(
        bandit_stats[
            [
                "movie_id",
                "alpha",
                "beta"
            ]
        ],
        on="movie_id",
        how="left"
    )


    candidates["alpha"] = (
        candidates["alpha"]
        .fillna(1.0)
    )


    candidates["beta"] = (
        candidates["beta"]
        .fillna(1.0)
    )


    # =====================================================
    # THOMPSON SAMPLING
    # =====================================================

    candidates["thompson_score"] = candidates.apply(

        lambda row: np.random.beta(

            max(
                float(row["alpha"]),
                0.01
            ),

            max(
                float(row["beta"]),
                0.01
            )

        ),

        axis=1
    )


    # =====================================================
    # GENRE MATCH
    # =====================================================

    liked_genres = user_liked_genres.get(
        user_id,
        []
    )


    def calculate_genre_score(genres):

        if not liked_genres:

            return 0.0


        movie_genres = str(
            genres
        ).split("|")


        if not movie_genres:

            return 0.0


        matches = sum(

            1

            for genre in movie_genres

            if genre in liked_genres

        )


        return matches / len(movie_genres)


    candidates["genre_score"] = (
        candidates["genres"]
        .apply(calculate_genre_score)
    )


    # =====================================================
    # FINAL SCORE
    # =====================================================

    candidates["final_score"] = (

        0.7 *
        candidates["thompson_score"]

        +

        0.3 *
        candidates["genre_score"]

    )


    # Сортировка
    candidates = candidates.sort_values(
        "final_score",
        ascending=False
    )


    result = candidates.head(k)


    # =====================================================
    # RESPONSE
    # =====================================================

    response = []


    for _, row in result.iterrows():

        response.append({

            "movie_id": int(
                row["movie_id"]
            ),

            "title": str(
                row["title"]
            ),

            "genres": str(
                row.get(
                    "genres",
                    ""
                )
            ),

            "thompson_score": float(
                row["thompson_score"]
            ),

            "genre_score": float(
                row["genre_score"]
            ),

            "final_score": float(
                row["final_score"]
            )

        })


    return response


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {

        "status": "ok",

        "service":
            "recommendation-system",

        "algorithm":
            "Thompson Sampling"

    }