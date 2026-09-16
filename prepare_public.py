import pandas as pd
import os

# Создаём папку для публичных данных
os.makedirs("public_data", exist_ok=True)

# Загружаем фильмы
movies = pd.read_csv("data/ml-25m/movies.csv")

# Сохраняем список фильмов
movies.to_csv("public_data/movies.csv", index=False)

# Загружаем рейтинги
ratings = pd.read_csv("data/ml-25m/ratings.csv")

# Reward:
# рейтинг 4.0 и выше = 1 (понравилось)
# рейтинг меньше 4.0 = 0 (не понравилось)
ratings["reward"] = (ratings["rating"] >= 4.0).astype(int)

# Статистика Thompson Sampling для каждого фильма
bandit_stats = ratings.groupby("movieId").agg(
    success=("reward", "sum"),
    total=("reward", "count")
).reset_index()

bandit_stats["alpha"] = bandit_stats["success"] + 1
bandit_stats["beta"] = (
    bandit_stats["total"] - bandit_stats["success"] + 1
)

bandit_stats.to_csv(
    "public_data/bandit_stats.csv",
    index=False
)

# Берём историю первых 1000 пользователей
user_ids = ratings["userId"].drop_duplicates().head(1000)

user_history = ratings[
    ratings["userId"].isin(user_ids)
][["userId", "movieId"]]

user_history.to_csv(
    "public_data/user_history.csv",
    index=False
)

print("Готово!")
print("Файлы созданы в папке public_data")
print("Количество фильмов:", len(movies))
print("Количество пользователей:", len(user_ids))
print("Количество историй:", len(user_history))
