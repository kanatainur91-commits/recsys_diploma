from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import pandas as pd
import numpy as np

from pathlib import Path

from sqlalchemy import text

from database import engine, create_tables


# =========================================================
# APPLICATION
# =========================================================

app = FastAPI(
    title="Интеллектуалды ұсыныстар жүйесі",
    description="Thompson Sampling негізіндегі фильмдер ұсынысы",
    version="1.0"
)


# =========================================================
# DATABASE
# =========================================================

create_tables()


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

if "movie_id" not in movies.columns:

    if "movieId" in movies.columns:

        movies = movies.rename(
            columns={
                "movieId": "movie_id"
            }
        )

    elif "id" in movies.columns:

        movies = movies.rename(
            columns={
                "id": "movie_id"
            }
        )


# =========================================================
# PREPARE BANDIT DATA
# =========================================================

if "movie_id" not in bandit_stats.columns:

    if "movieId" in bandit_stats.columns:

        bandit_stats = bandit_stats.rename(
            columns={
                "movieId": "movie_id"
            }
        )

    elif "id" in bandit_stats.columns:

        bandit_stats = bandit_stats.rename(
            columns={
                "id": "movie_id"
            }
        )


if "alpha" not in bandit_stats.columns:

    bandit_stats["alpha"] = 1.0


if "beta" not in bandit_stats.columns:

    bandit_stats["beta"] = 1.0


# =========================================================
# USER HISTORY
# =========================================================

user_history = {}


if not user_history_df.empty:

    if "user_id" not in user_history_df.columns:

        if "userId" in user_history_df.columns:

            user_history_df = user_history_df.rename(
                columns={
                    "userId": "user_id"
                }
            )


    if "movie_id" not in user_history_df.columns:

        if "movieId" in user_history_df.columns:

            user_history_df = user_history_df.rename(
                columns={
                    "movieId": "movie_id"
                }
            )


    if (
        "user_id" in user_history_df.columns
        and
        "movie_id" in user_history_df.columns
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
# LOAD DATABASE HISTORY
# =========================================================

def load_database_history():

    """
    Загружает уже сохранённую историю feedback
    из PostgreSQL.
    """

    if engine is None:
        return


    try:

        with engine.begin() as connection:

            result = connection.execute(
                text("""
                    SELECT
                        user_id,
                        movie_id,
                        reward
                    FROM feedback
                    ORDER BY timestamp
                """)
            )


            for row in result:

                uid = int(row.user_id)

                mid = int(row.movie_id)

                reward = int(row.reward)


                # -----------------------------------------
                # USER HISTORY
                # -----------------------------------------

                if uid not in user_history:

                    user_history[uid] = set()


                user_history[uid].add(mid)


                # -----------------------------------------
                # LIKED GENRES
                # -----------------------------------------

                if reward == 1:

                    movie_row = movies[
                        movies["movie_id"] == mid
                    ]


                    if not movie_row.empty:

                        genres = str(
                            movie_row.iloc[0].get(
                                "genres",
                                ""
                            )
                        )


                        if uid not in user_liked_genres:

                            user_liked_genres[uid] = []


                        for genre in genres.split("|"):

                            if (
                                genre
                                and
                                genre != "(no genres listed)"
                                and
                                genre not in user_liked_genres[uid]
                            ):

                                user_liked_genres[uid].append(
                                    genre
                                )

    except Exception as error:

        print(
            "Database history loading error:",
            error
        )


load_database_history()


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
# GET BANDIT DATA
# =========================================================

def get_bandit_stats():

    """
    Получает актуальные alpha/beta.

    Если PostgreSQL содержит данные —
    используем PostgreSQL.

    Для фильмов, которых ещё нет в PostgreSQL,
    используем начальные значения из CSV.
    """

    result = bandit_stats.copy()


    if engine is None:

        return result


    try:

        with engine.begin() as connection:

            db_result = connection.execute(
                text("""
                    SELECT
                        movie_id,
                        alpha,
                        beta
                    FROM bandit_stats
                """)
            )


            rows = db_result.fetchall()


        if not rows:

            return result


        db_stats = pd.DataFrame(
            [
                {
                    "movie_id": int(row.movie_id),
                    "alpha": float(row.alpha),
                    "beta": float(row.beta)
                }

                for row in rows
            ]
        )


        # ---------------------------------------------
        # PostgreSQL значения заменяют CSV значения
        # ---------------------------------------------

        result = result.drop(
            columns=[
                "alpha",
                "beta"
            ],
            errors="ignore"
        )


        result = result.merge(
            db_stats,
            on="movie_id",
            how="left"
        )


        result["alpha"] = (
            result["alpha"]
            .fillna(1.0)
        )


        result["beta"] = (
            result["beta"]
            .fillna(1.0)
        )


        return result


    except Exception as error:

        print(
            "Database bandit loading error:",
            error
        )

        return bandit_stats.copy()


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


    # =====================================================
    # USER HISTORY
    # =====================================================

    watched = user_history.get(
        user_id,
        set()
    )


    # =====================================================
    # CANDIDATES
    # =====================================================

    candidates = movies.copy()


    if watched:

        candidates = candidates[
            ~candidates["movie_id"].isin(watched)
        ]


    # =====================================================
    # BANDIT DATA
    # =====================================================

    current_bandit_stats = get_bandit_stats()


    candidates = candidates.merge(
        current_bandit_stats[
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

    candidates["score"] = candidates.apply(

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
    # SORT
    # =====================================================

    candidates = candidates.sort_values(
        "score",
        ascending=False
    )


    result = candidates.head(k)


    # =====================================================
    # LOG IMPRESSIONS
    # =====================================================

    if engine is not None:

        try:

            with engine.begin() as connection:

                for _, row in result.iterrows():

                    connection.execute(
                        text("""
                            INSERT INTO impressions (
                                user_id,
                                movie_id
                            )
                            VALUES (
                                :user_id,
                                :movie_id
                            )
                        """),
                        {
                            "user_id": user_id,
                            "movie_id": int(
                                row["movie_id"]
                            )
                        }
                    )

        except Exception as error:

            print(
                "Impression logging error:",
                error
            )


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


    # =====================================================
    # CHECK REWARD
    # =====================================================

    if reward not in [0, 1]:

        raise HTTPException(
            status_code=400,
            detail="Reward must be 0 or 1"
        )


    # =====================================================
    # CHECK MOVIE
    # =====================================================

    movie_exists = (
        movies["movie_id"] == item_id
    ).any()


    if not movie_exists:

        raise HTTPException(
            status_code=404,
            detail="Фильм табылмады"
        )


    # =====================================================
    # NEW ALPHA / BETA
    # =====================================================

    new_alpha = None

    new_beta = None


    # =====================================================
    # POSTGRESQL
    # =====================================================

    if engine is not None:

        try:

            with engine.begin() as connection:

                # -----------------------------------------
                # 1. Получаем текущий bandit state
                # -----------------------------------------

                current = connection.execute(
                    text("""
                        SELECT
                            alpha,
                            beta
                        FROM bandit_stats
                        WHERE movie_id = :movie_id
                    """),
                    {
                        "movie_id": item_id
                    }
                ).fetchone()


                # -----------------------------------------
                # 2. Если фильма ещё нет в БД,
                #    берём начальные значения из CSV
                # -----------------------------------------

                if current is None:

                    csv_row = bandit_stats[
                        bandit_stats["movie_id"] == item_id
                    ]


                    if not csv_row.empty:

                        old_alpha = float(
                            csv_row.iloc[0]["alpha"]
                        )

                        old_beta = float(
                            csv_row.iloc[0]["beta"]
                        )

                    else:

                        old_alpha = 1.0

                        old_beta = 1.0


                    new_alpha = (
                        old_alpha + reward
                    )

                    new_beta = (
                        old_beta + (1 - reward)
                    )


                    connection.execute(
                        text("""
                            INSERT INTO bandit_stats (
                                movie_id,
                                alpha,
                                beta
                            )
                            VALUES (
                                :movie_id,
                                :alpha,
                                :beta
                            )
                        """),
                        {
                            "movie_id": item_id,
                            "alpha": new_alpha,
                            "beta": new_beta
                        }
                    )


                # -----------------------------------------
                # 3. Если фильм уже есть в БД —
                #    атомарно увеличиваем alpha/beta
                # -----------------------------------------

                else:

                    connection.execute(
                        text("""
                            UPDATE bandit_stats
                            SET
                                alpha = alpha + :reward,
                                beta = beta + (1 - :reward),
                                updated_at = CURRENT_TIMESTAMP
                            WHERE movie_id = :movie_id
                        """),
                        {
                            "movie_id": item_id,
                            "reward": reward
                        }
                    )


                    updated = connection.execute(
                        text("""
                            SELECT
                                alpha,
                                beta
                            FROM bandit_stats
                            WHERE movie_id = :movie_id
                        """),
                        {
                            "movie_id": item_id
                        }
                    ).fetchone()


                    new_alpha = float(
                        updated.alpha
                    )

                    new_beta = float(
                        updated.beta
                    )


                # -----------------------------------------
                # 4. Сохраняем feedback
                # -----------------------------------------

                connection.execute(
                    text("""
                        INSERT INTO feedback (
                            user_id,
                            movie_id,
                            reward
                        )
                        VALUES (
                            :user_id,
                            :movie_id,
                            :reward
                        )
                    """),
                    {
                        "user_id": user_id,
                        "movie_id": item_id,
                        "reward": reward
                    }
                )


                # -----------------------------------------
                # 5. Обновляем impression
                # -----------------------------------------

                connection.execute(
                    text("""
                        UPDATE impressions
                        SET reward = :reward
                        WHERE id = (
                            SELECT id
                            FROM impressions
                            WHERE user_id = :user_id
                            AND movie_id = :movie_id
                            AND reward IS NULL
                            ORDER BY timestamp DESC
                            LIMIT 1
                        )
                    """),
                    {
                        "user_id": user_id,
                        "movie_id": item_id,
                        "reward": reward
                    }
                )


        except Exception as error:

            raise HTTPException(
                status_code=500,
                detail=f"Database error: {error}"
            )


    # =====================================================
    # LOCAL FALLBACK
    # =====================================================

    else:

        row_index = bandit_stats.index[
            bandit_stats["movie_id"] == item_id
        ]


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


        old_alpha = float(
            bandit_stats.at[
                idx,
                "alpha"
            ]
        )


        old_beta = float(
            bandit_stats.at[
                idx,
                "beta"
            ]
        )


        new_alpha = (
            old_alpha + reward
        )


        new_beta = (
            old_beta + (1 - reward)
        )


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
                and
                genre != "(no genres listed)"
                and
                genre not in user_liked_genres[user_id]
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


    # =====================================================
    # HISTORY
    # =====================================================

    watched = user_history.get(
        user_id,
        set()
    )


    candidates = movies.copy()


    if watched:

        candidates = candidates[
            ~candidates["movie_id"].isin(watched)
        ]


    # =====================================================
    # BANDIT DATA
    # =====================================================

    current_bandit_stats = get_bandit_stats()


    candidates = candidates.merge(
        current_bandit_stats[
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
        .apply(
            calculate_genre_score
        )
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


    # =====================================================
    # SORT
    # =====================================================

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

    database_status = "not connected"


    if engine is not None:

        try:

            with engine.begin() as connection:

                connection.execute(
                    text("SELECT 1")
                )


            database_status = "connected"


        except Exception:

            database_status = "error"


    return {

        "status": "ok",

        "service":
            "recommendation-system",

        "algorithm":
            "Thompson Sampling",

        "database":
            database_status

    }