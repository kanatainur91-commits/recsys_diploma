from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Зияткерлік ұсыныстар жүйесі",
    description="Thompson Sampling негізіндегі фильмдер ұсыныс жүйесі"
)


# =========================================================
# ДЕРЕКТЕРДІ ЖҮКТЕУ
# =========================================================

movies = pd.read_csv(
    "public_data/movies.csv"
)

bandit_stats = pd.read_csv(
    "public_data/bandit_stats.csv"
)

user_history_df = pd.read_csv(
    "public_data/user_history.csv"
)


# =========================================================
# ПАЙДАЛАНУШЫ ТАРИХЫ
# =========================================================

user_history = {}

for user_id, group in user_history_df.groupby("userId"):

    user_history[int(user_id)] = set(
        group["movieId"].astype(int)
    )


# =========================================================
# ПАЙДАЛАНУШЫНЫҢ ҰНАМДЫ ЖАНРЛАРЫ
# =========================================================

user_liked_genres = {}


# =========================================================
# БАСТЫ БЕТ
# =========================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    with open(
        "templates/index.html",
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


# =========================================================
# БАСТАПҚЫ ҰСЫНЫСТАР
# =========================================================

@app.get("/recommend")
def recommend(
    user_id: int,
    k: int = 10
):

    # Пайдаланушы бұрын көрген фильмдер
    watched = user_history.get(
        user_id,
        set()
    )


    # Бұрын көрілмеген фильмдерді аламыз
    available = bandit_stats[
        ~bandit_stats["movieId"].isin(watched)
    ].copy()


    if available.empty:

        raise HTTPException(
            status_code=404,
            detail="Бұл пайдаланушы үшін фильмдер табылмады"
        )


    # =====================================================
    # THOMPSON SAMPLING
    # =====================================================

    available["score"] = np.random.beta(
        available["alpha"],
        available["beta"]
    )


    # Ең жоғары Thompson score
    recommendations = available.nlargest(
        k,
        "score"
    )


    # Фильмдер туралы ақпаратты қосамыз
    result = recommendations.merge(
        movies,
        on="movieId"
    )


    return result[
        [
            "movieId",
            "title",
            "genres",
            "score"
        ]
    ].to_dict(
        orient="records"
    )


# =========================================================
# КЕРІ БАЙЛАНЫС
# =========================================================

@app.post("/feedback")
def feedback(
    user_id: int,
    item_id: int,
    reward: int
):

    # Reward тек 0 немесе 1 болуы керек
    if reward not in [0, 1]:

        raise HTTPException(
            status_code=400,
            detail="Reward 0 немесе 1 болуы керек"
        )


    # Фильмді іздеу
    index = bandit_stats.index[
        bandit_stats["movieId"] == item_id
    ]


    if len(index) == 0:

        raise HTTPException(
            status_code=404,
            detail="Фильм табылмады"
        )


    index = index[0]


    # =====================================================
    # THOMPSON SAMPLING ПАРАМЕТРЛЕРІН ЖАҢАРТУ
    # =====================================================

    if reward == 1:

        bandit_stats.loc[
            index,
            "alpha"
        ] += 1

    else:

        bandit_stats.loc[
            index,
            "beta"
        ] += 1


    # =====================================================
    # ПАЙДАЛАНУШЫ ТАРИХЫНА ҚОСУ
    # =====================================================

    if user_id not in user_history:

        user_history[user_id] = set()


    user_history[user_id].add(
        item_id
    )


    # =====================================================
    # ҰНАҒАН ФИЛЬМНІҢ ЖАНРЛАРЫН САҚТАУ
    # =====================================================

    if reward == 1:

        movie_info = movies[
            movies["movieId"] == item_id
        ]


        if not movie_info.empty:

            genres = str(
                movie_info.iloc[0]["genres"]
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
    # ЖАУАП
    # =====================================================

    return {

        "message":
            "Баға сәтті сақталды",

        "user_id":
            user_id,

        "item_id":
            item_id,

        "reward":
            reward,

        "alpha":
            float(
                bandit_stats.loc[
                    index,
                    "alpha"
                ]
            ),

        "beta":
            float(
                bandit_stats.loc[
                    index,
                    "beta"
                ]
            )

    }


# =========================================================
# ҚОРЫТЫНДЫ ҰСЫНЫСТАР
# =========================================================

@app.get("/final-recommend")
def final_recommend(
    user_id: int,
    k: int = 5
):

    # =====================================================
    # ПАЙДАЛАНУШЫ ТАРИХЫ
    # =====================================================

    watched = user_history.get(
        user_id,
        set()
    )


    # =====================================================
    # ҰНАМДЫ ЖАНРЛАР
    # =====================================================

    liked_genres = user_liked_genres.get(
        user_id,
        []
    )


    # =====================================================
    # ҚОЛЖЕТІМДІ ФИЛЬМДЕР
    # =====================================================

    available = bandit_stats[
        ~bandit_stats["movieId"].isin(watched)
    ].copy()


    if available.empty:

        raise HTTPException(
            status_code=404,
            detail="Жаңа фильмдер табылмады"
        )


    # =====================================================
    # THOMPSON SAMPLING
    # =====================================================

    available["thompson_score"] = np.random.beta(
        available["alpha"],
        available["beta"]
    )


    # =====================================================
    # ФИЛЬМ АҚПАРАТЫН ҚОСУ
    # =====================================================

    result = available.merge(
        movies,
        on="movieId"
    )


    # =====================================================
    # ЖАНР СӘЙКЕСТІГІ
    # =====================================================

    def calculate_genre_match(
        genres
    ):

        if not liked_genres:

            return 0


        movie_genres = str(
            genres
        ).split("|")


        matches = 0


        for genre in movie_genres:

            if genre in liked_genres:

                matches += 1


        return matches


    result["genre_match"] = result[
        "genres"
    ].apply(
        calculate_genre_match
    )


    # =====================================================
    # ЖАНР СӘЙКЕСТІГІН 0-1 АРАЛЫҒЫНА АУЫСТЫРУ
    # =====================================================

    if liked_genres:

        result["genre_score"] = (
            result["genre_match"]
            /
            max(len(liked_genres), 1)
        )

    else:

        result["genre_score"] = 0


    # =====================================================
    # ҚОРЫТЫНДЫ ҰПАЙ
    #
    # 70% — Thompson Sampling
    # 30% — пайдаланушы жанрларының сәйкестігі
    # =====================================================

    result["final_score"] = (

        result["thompson_score"] * 0.7

        +

        result["genre_score"] * 0.3

    )


    # =====================================================
    # ЕҢ ЖОҒАРЫ ҚОРЫТЫНДЫ ҰПАЙЛАР
    # =====================================================

    recommendations = result.nlargest(
        k,
        "final_score"
    )


    return recommendations[
        [
            "movieId",
            "title",
            "genres",
            "thompson_score",
            "genre_score",
            "final_score"
        ]
    ].to_dict(
        orient="records"
    )


# =========================================================
# СЕРВЕРДІ ТЕКСЕРУ
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "message": "Сервер жұмыс істеп тұр"
    }

