# Music Popularity Prediction

This repository contains a data science project that aims to predict the popularity of music using machine learning techniques.

## Dataset

This project uses the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) available on Kaggle. This dataset contains information about Spotify tracks over a range of 125 different genres. Each track has several audio features associated with it, such as popularity, explicitness, danceability, energy, key, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, and time signature.

You can download the dataset from the Kaggle website and use it to follow along with the analysis in this project.

## Overview

This repository contains a data science project that aims to predict the popularity of music using machine learning techniques. The project is a binary classification problem where the goal is to predict whether a song will be popular or not. The dataset used in this project is imbalanced, meaning that one class is significantly more common than the other.

The project consists of three main parts: Data Cleaning, Exploratory Data Analysis, and Model Building.

### Data Cleaning

In the [Data Cleaning](https://github.com/diivien/Music-Popularity-Prediction/blob/master/Data%20Cleaning.ipynb) notebook, I clean and preprocess the data to prepare it for analysis. This involves several steps such as:

- Removing unique columns
- Dropping null values
- Dropping duplicated rows (same artists and track name)
- Dropping artists and track name columns
- Dropping invalid tempo and time signature according to Spotify API
- Saving the cleaned dataset into a CSV file

To get started with the data cleaning process, you can follow the instructions in the Data Cleaning notebook. This will guide you through the steps involved in cleaning and preprocessing the data.

### Exploratory Data Analysis

In the [Exploratory Data Analysis](https://github.com/diivien/Music-Popularity-Prediction/blob/master/Exploratory%20Data%20Analysis.ipynb) notebook, we explore the data and gain insights into the relationships between the features and the target variable. This involves generating various visualizations such as:

- Correlation heatmaps to examine the relationships between pairs of continuous features
- Histograms to check the distribution of continuous features
- Bar charts to visualize categorical features
- Scatter plots to examine the relationships between pairs of continuous features
- Box plots to examine the distribution of continuous features by category
- Stacked bar charts to visualize conditional distributions

These visualizations help us understand the data better and inform our decisions when building machine learning models.

To get started with the exploratory data analysis process, you can follow the instructions in the Exploratory Data Analysis notebook. This will guide you through the steps involved in exploring and visualizing the data.

### Model Building

In the [Model Building](https://github.com/diivien/Music-Popularity-Prediction/blob/master/Model%20Building.ipynb) notebook, I build and evaluate machine learning models to predict music popularity. The models used in this analysis include Linear SVC, Random Forest Classifier, LightGBM, and CatBoost. As part of this process, I perform several preprocessing steps such as scaling the data using a MinMax scaler and encoding categorical variables using a target encoder. I also use SMOTE-NC in an imbalanced-learn pipeline to prevent data leakage.

To tune the hyperparameters of our models, I use Optuna for multi-objective optimization and generate a Pareto front plot to determine the best hyperparameters.

To evaluate the performance of our models, I use several metrics that are appropriate for imbalanced datasets, such as F1 score, balanced accuracy, and PR AUC.

To get started with the model building process, you can follow the instructions in the Model Building notebook. This will guide you through the steps involved in building and evaluating machine learning models to predict music popularity.

## Future Work

I am currently working on several improvements and extensions to this project. Some include:

- Testing a neural network classifier to see if it can improve the accuracy of our predictions
- Deploying an app on Gradio to make it easier for users to interact with our models and make predictions


## Citations

If you use any of the following libraries in your project, please cite them as follows:

- imbalanced-learn: Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning. Journal of Machine Learning Research, 18(17), 1-5.
- Matplotlib: Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90-95.
- Seaborn: Waskom, M., Botvinnik, O., O’Kane, D., Hobson, P., Lukauskas, S., Gemperline, D. C., ... & de Ruiter, J. (2021). seaborn: statistical data visualization. Journal of Open Source Software, 6(60), 3021.
- Joblib: Buitinck, L., Louppe, G., Blondel, M., Pedregosa, F., Mueller, A., Grisel, O., ... & Duchesnay, E. (2013). API design for machine learning software: experiences from the scikit-learn project. arXiv preprint arXiv:1309.0238.
- Feature-engine: Sole-Ribalta A. (2020) Feature-engine: A Python Package for Feature Engineering and Preprocessing in Machine Learning. In: Martínez-Villaseñor L., Batyrshin I., Mendoza O., Kuri-Morales Á. (eds) Advances in Artificial Intelligence - IBERAMIA 2020. IBERAMIA 2020. Lecture Notes in Computer Science, vol 12422. Springer, Cham.
- LightGBM: Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems.
- CatBoost: Prokhorenkova L.O., Gusev G.L., Vorobev A.V., Dorogush A.V., Gulin A.A.(2018). CatBoost: unbiased boosting with categorical features. Advances in Neural Information Processing Systems.
- Category Encoders: Micci-Barreca D (2001) A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. ACM SIGKDD Explorations Newsletter 3(1):27–32
- NumPy: Harris CR et al.(2020) Array programming with NumPy. Nature 585(7825):357–362
- SDV (Synthetic Data Vault): Patki N et al.(2016) The Synthetic Data Vault. IEEE International Conference on Data Science and Advanced Analytics
- Optuna: Akiba T et al.(2019) Optuna: A Next-generation Hyperparameter Optimization Framework. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining
- PyTorch: Paszke A et al.(2019) PyTorch: An Imperative Style High-performance Deep Learning Library. Advances in Neural Information Processing Systems
- SciKeras: Varma P et al.(2020) SciKeras: a high-level Scikit-Learn compatible API for TensorFlow's Keras module
