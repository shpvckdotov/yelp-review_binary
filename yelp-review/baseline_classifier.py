import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TfidfLogRegClassifier:
    """
    Класс для бинарной классификации текстов с использованием TF-IDF + Logistic Regression
    """

    def __init__(
        self,
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
        C=1.0,
        class_weight=None,
        random_state=42,
    ):
        """
        Инициализация классификатора

        Args:
            max_features (int): Максимальное количество признаков TF-IDF
            ngram_range (tuple): Диапазон n-грамм
            stop_words (str/list): Стоп-слова
            C (float): Параметр регуляризации для логистической регрессии
            class_weight (dict/str): Веса классов
            random_state (int): Seed для воспроизводимости
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state

        # Создаем pipeline
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=self.max_features,
                        ngram_range=self.ngram_range,
                        stop_words=self.stop_words,
                        lowercase=True,
                        analyzer="word",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=self.C,
                        class_weight=self.class_weight,
                        random_state=self.random_state,
                        max_iter=1000,
                        solver="liblinear",
                    ),
                ),
            ]
        )

        self.is_trained = False

    def fit(self, X_train, y_train, sample_weight=None):
        """
        Обучение модели

        Args:
            X_train (list/array): Тексты для обучения
            y_train (array): Метки классов
            sample_weight (array): Веса примеров

        Returns:
            self: Обученная модель
        """

        if sample_weight is not None:
            self.pipeline.fit(X_train, y_train, clf__sample_weight=sample_weight)
        else:
            self.pipeline.fit(X_train, y_train)

        self.is_trained = True

        # Сохраняем информацию о классах
        self.classes_ = self.pipeline.named_steps["clf"].classes_

        return self

    def predict(self, X):
        """
        Предсказание классов

        Args:
            X (list/array): Тексты для предсказания

        Returns:
            array: Предсказанные классы
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите fit().")

        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """
        Предсказание вероятностей

        Args:
            X (list/array): Тексты для предсказания

        Returns:
            array: Вероятности классов [P(class=0), P(class=1)]
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите fit().")

        return self.pipeline.predict_proba(X)

    def predict_log_proba(self, X):
        """
        Предсказание логарифмических вероятностей

        Args:
            X (list/array): Тексты для предсказания

        Returns:
            array: Логарифмические вероятности классов
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите fit().")

        return self.pipeline.named_steps["clf"].predict_log_proba(
            self.pipeline.named_steps["tfidf"].transform(X)
        )

    def get_feature_names(self):
        """
        Получение списка фичей (слов/n-грамм)

        Returns:
            list: Список фичей
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите fit().")

        return self.pipeline.named_steps["tfidf"].get_feature_names_out()

    def save(self, filepath):
        """
        Сохранение модели на диск

        Args:
            filepath (str): Путь для сохранения модели
        """
        joblib.dump(self.pipeline, filepath)

    @classmethod
    def load(cls, filepath):
        """
        Загрузка модели с диска

        Args:
            filepath (str): Путь к сохраненной модели

        Returns:
            TfidfLogRegClassifier: Загруженная модель
        """
        classifier = cls()
        classifier.pipeline = joblib.load(filepath)
        classifier.is_trained = True
        classifier.classes_ = classifier.pipeline.named_steps["clf"].classes_
        return classifier
