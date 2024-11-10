# Twitter Sentiment Analysis

## Overview
This project implements a machine learning solution for analyzing sentiment in tweets, classifying them as either normal (0) or negative/hate speech (1). The project uses various NLP techniques and machine learning models to achieve this classification.

## Features
- Text preprocessing and cleaning
- Multiple feature extraction methods:
  - Bag of Words (BoW)
  - TF-IDF (Term Frequency-Inverse Document Frequency)
  - Word2Vec embeddings
  - Doc2Vec embeddings
- Various machine learning models:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - XGBoost (with parameter tuning)
- Visualization of results using WordCloud
- Hashtag analysis

## Requirements
```
numpy
pandas
scikit-learn
nltk
gensim
xgboost
seaborn
matplotlib
wordcloud
tqdm
```

## Dataset
The project uses two main datasets:
- `train_E6oV3lV.csv`: Training dataset with labeled tweets
- `test_tweets_anuFYb8.csv`: Test dataset for predictions

Expected format:
- Training data: Contains 'tweet' and 'label' columns
- Test data: Contains 'tweet' column

## Project Structure
1. **Data Preprocessing**
   - Text cleaning (removing Twitter handles, punctuation, numbers)
   - Word normalization
   - Short word removal
   - Text tokenization and stemming

2. **Feature Engineering**
   - Bag of Words vectorization
   - TF-IDF vectorization
   - Word2Vec embeddings
   - Doc2Vec embeddings

3. **Model Implementation**
   - Logistic Regression
   - SVM with linear kernel
   - Random Forest (400 estimators)
   - XGBoost with parameter tuning

4. **Visualization**
   - WordCloud for all tweets
   - Separate WordClouds for normal and negative tweets
   - Hashtag frequency analysis

## Model Performance
The project evaluates models using F1 score. Different combinations of features and models are tested:
- Bag of Words features
- TF-IDF features
- Word2Vec embeddings
- Doc2Vec embeddings

## XGBoost Parameter Tuning
Detailed parameter tuning is performed for XGBoost including:
- max_depth and min_child_weight
- subsample and colsample
- learning rate (eta)

## Output
The models generate prediction files in CSV format:
- `sub_lreg_bow.csv`
- `sub_svm_bow.csv`
- `sub_rf_bow.csv`
- `sub_xgb_tfidf.csv`
- `sub_xgb_w2v_finetuned.csv`

## Usage
1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Prepare your data in the required format

3. Run the preprocessing pipeline:
```python
python twitter_sentiment_analysis.py
```

## Best Practices
- Use cross-validation for reliable model evaluation
- Tune hyperparameters for optimal performance
- Consider class imbalance in the dataset
- Regularly update word embeddings with new data

## Future Improvements
- Implement deep learning models (LSTM, BERT)
- Add real-time tweet processing
- Enhance preprocessing pipeline
- Add multi-language support
- Implement ensemble methods
