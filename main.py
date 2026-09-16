import pandas as pd
import numpy as np


# ==========================================
# 1. ДЕРЕКТЕРДІ ЖҮКТЕУ
# ==========================================

# Рейтингтерді жүктеу
ratings = pd.read_csv("data/ml-25m/ratings.csv")

# Фильмдер туралы ақпаратты жүктеу
movies = pd.read_csv("data/ml-25m/movies.csv")


# ==========================================
# 2. REWARD ЕСЕПТЕУ
# ==========================================

# Егер рейтинг 4.0 немесе одан жоғары болса:
# reward = 1
# Әйтпесе reward = 0
ratings["reward"] = (ratings["rating"] >= 4.0).astype(int)


# ==========================================
# 3. РЕЙТИНГТЕР МЕН ФИЛЬМДЕРДІ БІРІКТІРУ
# ==========================================

data = ratings.merge(
    movies,
    on="movieId"
)

print("Деректер сәтті біріктірілді!")
print()

print("Жолдар саны:", len(data))
print()

print("Алғашқы 5 жол:")
print(data.head())

print()

print("Бағандар:")
print(data.columns.tolist())


# ==========================================
# 4. TOP-POPULAR ФИЛЬМДЕР
# ==========================================

popular_movies = (
    data.groupby(["movieId", "title"])
    .agg(
        ratings_count=("rating", "count"),
        average_rating=("rating", "mean")
    )
    .sort_values(
        "ratings_count",
        ascending=False
    )
    .head(10)
    .reset_index()
)


print()
print("Ең танымал 10 фильм:")
print(popular_movies)


# ==========================================
# 5. TOP-POPULAR RECOMMENDATION
# ==========================================

def recommend_popular(k=10):

    recommendations = popular_movies.head(k)

    print()
    print(f"Ұсынылатын фильмдер саны: {k}")
    print()

    for i, row in enumerate(
        recommendations.itertuples(),
        start=1
    ):
        print(f"{i}. {row.title}")

    return recommendations


# ==========================================
# 6. THOMPSON SAMPLING ҮШІН СТАТИСТИКА
# ==========================================

bandit_stats = (
    data.groupby("movieId")
    .agg(
        success=("reward", "sum"),
        total=("reward", "count")
    )
)


# Alpha:
# оң reward саны + 1
bandit_stats["alpha"] = (
    bandit_stats["success"] + 1
)


# Beta:
# теріс reward саны + 1
bandit_stats["beta"] = (
    bandit_stats["total"]
    - bandit_stats["success"]
    + 1
)


print()
print("Thompson Sampling параметрлері:")
print(
    bandit_stats.head(10)
)


# ==========================================
# 7. THOMPSON SAMPLING
# ==========================================

def recommend_thompson(k=10):

    # Beta(alpha, beta)
    # үлестірімінен кездейсоқ мән алу
    bandit_stats["sample"] = np.random.beta(
        bandit_stats["alpha"],
        bandit_stats["beta"]
    )

    # Ең жоғары sample мәндері бар
    # фильмдерді таңдау
    recommendations = (
        bandit_stats
        .sort_values(
            "sample",
            ascending=False
        )
        .head(k)
        .reset_index()
    )


    # Фильм атауын қосу
    recommendations = recommendations.merge(
        movies[["movieId", "title"]],
        on="movieId"
    )


    print()
    print(
        f"Thompson Sampling арқылы "
        f"ұсынылатын фильмдер саны: {k}"
    )
    print()


    for i, row in enumerate(
        recommendations.itertuples(),
        start=1
    ):
        print(f"{i}. {row.title}")


    return recommendations


# ==========================================
# 8. НӘТИЖЕНІ КӨРСЕТУ
# ==========================================

print()
print("==============================")
print("TOP-POPULAR ҰСЫНЫСТАР")
print("==============================")

recommend_popular(10)


print()
print("==============================")
print("THOMPSON SAMPLING ҰСЫНЫСТАРЫ")
print("==============================")

recommend_thompson(10)