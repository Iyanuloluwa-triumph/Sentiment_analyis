# Demystifying Logistic Regression for Sentiment Analysis ✅

## Overview
This project explains and demonstrates logistic regression for sentiment analysis (binary classification) using simple, interpretable features derived from movie review text. The notebook `Sentiment Analysis/log_reg.ipynb` walks through theory, visualizations, and a full modeling pipeline: preprocessing, feature extraction, training, hyperparameter tuning, evaluation, and model comparison against Naive Bayes and Decision Tree classifiers. Finally, the project is containerized with Docker for reproducibility.

## What you'll learn 💡
- Intuition behind logistic regression and why the sigmoid maps linear outputs to probabilities
- How to implement logistic regression using gradient descent (from-scratch educational implementation)
- Building a simple sentiment feature set (bias + positive/negative token counts)
- How to preprocess text thoroughly: cleaning, stopword removal, stemming, log-scaling, and standardization
- How to compare models (Logistic Regression, Naive Bayes, Decision Tree) and perform hyperparameter tuning
- Packaging the experiment into a Docker container for reproducibility

## Files
- `Sentiment Analysis/log_reg.ipynb` — Notebook that contains: logistic regression derivation, visualizations, preprocessing steps, model training and comparison, hyperparameter tuning, and evaluation.
- `Sentiment Analysis/utils.py` — Helper functions: `preprocess_text`, `build_frequency_dict`, `extract_features`.
- `IDMB_movie_dataset/MOVIES.csv` — IMDB movie dataset (pre-downloaded) used for experiments.
- `Dockerfile` — Builds container to run the notebook and reproduce experiments.
- `requirements.txt` — Python dependencies.

## How the notebook is organized 🔧
1. Introduction and visual intuition: sigmoid, cost, and gradients
2. Data loading: load IMDB data from `IDMB_movie_dataset/MOVIES.csv` and/or NLTK `movie_reviews`
3. Preprocessing:
   - Clean non-alpha characters
   - Lowercase
   - Remove stopwords
   - Stem words (PorterStemmer)
   - Build frequency dictionary using only training data (no leakage)
   - Log-scale and Standardize numeric features
   - Labeling note: when a dataset lacks explicit sentiment labels (e.g., `IDMB_movie_dataset/MOVIES.csv`), the notebook derives a proxy sentiment using `vote_average` thresholds (>=7 => positive, <=5 => negative) and drops neutral examples to reduce label noise.
4. Model implementations:
   - Educational logistic regression (gradient descent) for intuition and diagnostics
   - Scikit-learn logistic regression, Multinomial Naive Bayes, DecisionTreeClassifier for reliable baselines
5. Hyperparameter tuning using GridSearchCV for each model
6. Final evaluation: accuracy, precision, recall, F1, confusion matrix, ROC/AUC
7. Wrap-up: model comparison and takeaways

## How to reproduce locally 🧭
1. Create and activate a Python environment (suggested Python 3.8+)
2. Install requirements:

   pip install -r requirements.txt

3. Open and run `Sentiment Analysis/log_reg.ipynb` in Jupyter or VS Code.

## Containerized reproduction (Docker) 🐳
1. Build the image:

   docker build -t sentiment-demo:latest .

2. Run the container and start Jupyter Lab (port mapping example):

   docker run --rm -p 8888:8888 -v $(pwd):/workspace sentiment-demo:latest

3. Open the notebook from the Jupyter UI and re-run experiments.

## Notes and Tips ⚠️
- The from-scratch logistic regression is intentionally simple for pedagogy; use scikit-learn for production-ready training and tuning.
- Always avoid data leakage: build dictionaries and feature scalers on training data only.
- Use log1p on token counts to reduce skewness.

---
