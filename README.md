NLP Sentiment Analysis - Union Budget
Student Information

    Name: Kashish Negi
    Roll No: 2301730287
    Course: BTECH CSE (AI-ML)
    Section: E
    Semester: 5

Project Overview

This project performs sentiment analysis on Union Budget related data using Natural Language Processing (NLP) techniques and machine learning algorithms.
Project Setup
Prerequisites

    Python 3.7 or higher
    Jupyter Notebook or JupyterLab

Required Libraries

Install the following Python packages:

pip install pandas numpy matplotlib seaborn nltk scikit-learn textblob wordcloud tweepy

Detailed Dependencies

    pandas - Data manipulation and analysis
    numpy - Numerical computing
    matplotlib - Data visualization
    seaborn - Statistical data visualization
    nltk - Natural Language Toolkit for text processing
    scikit-learn - Machine learning algorithms and tools
    textblob - Text processing and sentiment analysis
    wordcloud - Word cloud generation
    tweepy - Twitter API integration (if needed)

NLTK Data Downloads

After installing nltk, run the following in Python:

import nltk
nltk.download('stopwords')
nltk.download('punkt')

Project Structure

NLP-SENTIMENTPROJECT/
│
├── union budeget sentiment analysis (1).ipynb    # Main Jupyter notebook
├── MRFS_1_Union_Budget.csv                       # Dataset
└── README.md                                      # Project documentation

How to Run

Clone or navigate to the project directory:

cd /Users/khushi_ydv/NLP-SENTIMENTPROJECT

Install all required dependencies:

pip install pandas numpy matplotlib seaborn nltk scikit-learn textblob wordcloud tweepy

Launch Jupyter Notebook:

jupyter notebook

    Open the notebook:
        Open union budeget sentiment analysis (1).ipynb
        Run all cells sequentially

Features

    Data preprocessing and cleaning
    Text analysis using NLTK
    Sentiment classification (Positive/Negative/Neutral)
    Emotion detection (Anger, Fear, Sad, Hatred, Love, Happy)
    Visualization using matplotlib and seaborn
    Word cloud generation
    Machine learning model training (Logistic Regression)
    TF-IDF vectorization
    Bigram analysis
    Model evaluation with classification reports and confusion matrices

Dataset

The project uses MRFS_1_Union_Budget.csv containing Union Budget related text data for sentiment analysis.
Models & Techniques Used

    Text Preprocessing: Stopword removal, stemming (SnowballStemmer)
    Feature Extraction: TF-IDF Vectorizer, Count Vectorizer
    Machine Learning: Logistic Regression
    Sentiment Analysis: TextBlob
    Label Encoding: For emotion categories

Contact
For any queries regarding this project, please contact Kashish Negi (Roll No: 2301730287).
