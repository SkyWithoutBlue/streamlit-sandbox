import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split

class DataManager:
    """
    Класс для генерации, предварительной обработки и разделения наборов данных.
    """
    @staticmethod
    def generate_dataset(dataset_type: str, n_samples: int, noise: float):
        """
        Генерирует двумерные наборы данных для классификации.
        """
        if "Полумесяцы" in dataset_type:
            X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        elif "Круги" in dataset_type:
            X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
        else: # Облака / Блоки (Мультиклассовая классификация)
            X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=noise * 5, random_state=42)
            # Приведение меток к бинарным для совместимости с базовой моделью
            y = np.where(y == 2, 1, y) 
        return X, y

    @staticmethod
    def prepare_data(X: np.ndarray, y: np.ndarray, test_size: float):
        """
        Разделяет данные на обучающую и тестовую выборки.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        return X_train, X_test, y_train, y_test

    @staticmethod
    def get_metrics_summary(df: pd.DataFrame):
        """
        Считает распределение классов в наборе данных.
        """
        summary = df["target"].value_counts().to_dict()
        return summary
