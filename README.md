# NASDAQ Prediction with LSTM
This repository contains a Jupyter notebook and Python scripts that use an LSTM neural network to predict future values of the NASDAQ index based on historical data.

## Files

**NASDAQ-data-analysis-LSTM.ipynb :** A Jupyter notebook that contains the data analysis, model training, and prediction code. The notebook includes detailed explanations and visualizations of the data analysis and model training process.

**training.py :** A Python script that loads the NASDAQ dataset, trains the LSTM model, and saves the trained model to a file.

**evaluation.py :** A Python script that loads the trained model, evaluates its performance on the test data, and generates a visualization of the predicted vs. actual values.

**dag.py :** A Python script that defines an Airflow DAG for running the training and evaluation tasks.

## Requirements
The following packages are required to run the code in this repository:

- Python 3.6+
- TensorFlow 2.0+
- Pandas
- Numpy
- Matplotlib
- Scikit-learn
- Airflow (for running the DAG)


## Contributing
Contributions to this repository are welcome. If you find a bug or have a suggestion for improvement, please open an issue or submit a pull request.
