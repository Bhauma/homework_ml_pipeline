# Импорт необходимых модулей для работы с датами и создания DAG
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# Определяем общие параметры для DAG
default_args = {
    'owner': 'airflow',                         # Владелец DAG
    'start_date': datetime(2025, 1, 1),           # Дата первого запуска DAG
    'retries': 1,                               # Количество повторных запусков в случае ошибки
    'retry_delay': timedelta(minutes=5)         # Задержка между повторными попытками
}

# Создаем DAG с уникальным именем и описанием
with DAG(
    'ml_pipeline',                             # Уникальное имя DAG
    default_args=default_args,
    description='Пример интеграции AirFlow и MLFlow для обучения модели',
    schedule_interval='@daily',                # Расписание запуска: ежедневно
    catchup=False                              # Не выполнять пропущенные запуски
) as dag:

    # Создаем задачу, которая запускает скрипт обучения модели через BashOperator
    train_model = BashOperator(
        task_id='run_train_script',            # Уникальный идентификатор задачи
        # Запускаем Python-скрипт, который находится в директории mlflow_projects
        bash_command='python mlflow_projects/train.py'
    )

    # Определяем порядок выполнения задач (в данном случае одна задача)
    train_model
