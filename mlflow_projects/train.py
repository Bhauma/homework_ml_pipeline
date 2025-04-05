"""
Скрипт для обучения модели и логирования эксперимента с использованием MLFlow.
В данном примере используется датасет ирисов из библиотеки scikit-learn и модель RandomForestRegressor.
"""

# Импорт необходимых библиотек
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math  # для вычисления квадратного корня

def main():
    # Устанавливаем название эксперимента в MLFlow.
    # Если эксперимент с таким названием отсутствует, он будет создан.
    mlflow.set_experiment("Example_MLFlow_AirFlow")

    # Начинаем новый запуск эксперимента
    with mlflow.start_run():
        # Задаем гиперпараметры модели
        n_estimators = 100
        max_depth = 5

        # Логируем гиперпараметры в MLFlow для последующего анализа
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Загрузка датасета диабета
        data = load_iris()
        # Делим данные на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42)

        # Создаем и обучаем модель случайного леса
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Предсказываем значения на тестовой выборке
        predictions = model.predict(X_test)
        
        # Вычисляем MSE без использования параметра squared
        mse = mean_squared_error(y_test, predictions)
        # Вычисляем RMSE как квадратный корень из MSE
        rmse = math.sqrt(mse)
        
        # Логируем метрику RMSE в MLFlow
        mlflow.log_metric("rmse", rmse)

        # Логирование обученной модели с передачей примера входных данных (первые данные из X_test)
        mlflow.sklearn.log_model(model, "model", input_example=X_test[0:1])


        # Выводим результат в консоль
        print(f"Обучение завершено. RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()
