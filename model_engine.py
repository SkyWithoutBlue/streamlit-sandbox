import numpy as np
import matplotlib.pyplot as plt
import time
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

# Попытка импорта TensorFlow / Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam, SGD
    keras_available = True
except ImportError:
    keras_available = False

class ModelEngine:
    """
    Класс для управления жизненным циклом нейросетевых моделей:
    создание, пошаговое обучение, визуализация границ разделения классов и экспорт.
    """
    def __init__(self, engine_type: str, hidden_layers: tuple, activation: str, learning_rate: float):
        self.engine_type = engine_type if (engine_type != "TensorFlow/Keras" or keras_available) else "Scikit-learn (MLP)"
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.model = None
        self.is_trained = False
        
        # Словарь маппинга активаций для Keras
        self.keras_activation_map = {
            "logistic": "sigmoid",
            "relu": "relu",
            "tanh": "tanh",
            "identity": "linear"
        }

    def initialize_model(self, input_dim: int):
        """
        Создает экземпляр модели на основе выбранного движка и параметров.
        """
        if self.engine_type == "TensorFlow/Keras" and keras_available:
            # 1. Построение Keras-модели на основе переданной структуры слоев
            self.model = Sequential()
            # Первый скрытый слой
            self.model.add(Dense(
                self.hidden_layers[0], 
                activation=self.keras_activation_map.get(self.activation, "relu"), 
                input_shape=(input_dim,)
            ))
            
            # Последующие скрытые слои
            for neurons in self.hidden_layers[1:]:
                if neurons > 0:
                    self.model.add(Dense(
                        neurons, 
                        activation=self.keras_activation_map.get(self.activation, "relu")
                    ))
            
            # Выходной бинарный слой
            self.model.add(Dense(1, activation="sigmoid"))
            
            # Компиляция
            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
            
        else:
            # 2. Инициализация Scikit-learn MLPClassifier
            self.model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layers,
                activation=self.activation,
                learning_rate_init=self.learning_rate,
                max_iter=1,
                random_state=42,
                warm_start=True
            )

    def train_epoch(self, X_train: np.ndarray, y_train: np.ndarray, classes: np.ndarray = None):
        """
        Выполняет одну эпоху/итерацию обучения и возвращает текущую ошибку и точность.
        """
        if self.engine_type == "TensorFlow/Keras" and keras_available:
            history = self.model.fit(X_train, y_train, epochs=1, batch_size=4, verbose=0)
            loss = history.history['loss'][0]
            acc = history.history['accuracy'][0]
        else:
            self.model.partial_fit(X_train, y_train, classes=classes)
            loss = self.model.loss_
            acc = self.model.score(X_train, y_train)
            
        return loss, acc

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Рассчитывает расширенные метрики качества на тестовой выборке.
        """
        if self.engine_type == "TensorFlow/Keras" and keras_available:
            preds_prob = self.model.predict(X_test, verbose=0)
            preds = np.where(preds_prob >= 0.5, 1, 0).flatten()
        else:
            preds = self.model.predict(X_test)
            
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        conf_matrix = confusion_matrix(y_test, preds)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": conf_matrix,
            "predictions": preds
        }

    def get_decision_boundary(self, X: np.ndarray, resolution: int = 50):
        """
        Генерирует координатную сетку для отрисовки границ решения.
        """
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        if self.engine_type == "TensorFlow/Keras" and keras_available:
            Z = self.model.predict(grid_points, verbose=0).flatten()
        else:
            Z = self.model.predict_proba(grid_points)[:, 1]
            
        Z = Z.reshape(xx.shape)
        return xx, yy, Z

    def predict_point(self, point: np.ndarray):
        """
        Делает индивидуальный прогноз для отдельной точки.
        """
        if self.engine_type == "TensorFlow/Keras" and keras_available:
            prob = float(self.model.predict(point, verbose=0)[0][0])
            pred = 1 if prob >= 0.5 else 0
            conf = prob if pred == 1 else (1 - prob)
        else:
            probs = self.model.predict_proba(point)[0]
            pred = int(self.model.predict(point)[0])
            conf = float(probs[pred])
            
        return pred, conf

    def save_model(self, file_path: str):
        """
        Экспортирует обученную модель в файл.
        """
        if self.engine_type == "TensorFlow/Keras" and keras_available:
            self.model.save(file_path + ".h5")
        else:
            joblib.dump(self.model, file_path + ".pkl")
            
    @staticmethod
    def is_keras_supported():
        return keras_available
