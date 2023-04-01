from datetime import datetime, timedelta

from airflow import DAG

from airflow.operators.python import PythonOperator

from training import training
from evaluation import evaluate

default_args = {
    'owner': 'NASDAQ-MODEL',
    'retries': 5,
    'retry_delay': timedelta(minutes=2)
}
with DAG(
        dag_id='ml_project',
        default_args=default_args,
        description='ML project',
        schedule_interval='* * * * *',
        start_date=datetime(2020, 1, 6)
) as dag:
    train_model = PythonOperator(
        task_id='train_model',
        python_callable=training
    )

    evaluate_model = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate
    )

 
    train_model >> evaluate_model
