import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import sys
import pickle
import re

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(database_filepath):
    """function for loading data from database (SQLite) and creating X and Y for use in the machine learning models
    Input:  filepath of database 
    Output: features(X), targets(Y) and category names """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table ('DisasterData', con=engine)
    category_names = df.columns[-36:]
    #Creating X and Y for use in the machine learning models - where X is the input and Y is the output
    X = df['message']
    y = df[category_names]
    return X,y, category_names

def tokenize(text):
    """function for tokenize text, remove stop words and reduces words to root form
    Input: text to tokenize
    Output: cleaned text """
    #Normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    #Tokenize text
    words = nltk.word_tokenize(text)
    #Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    #Reduce to root form
    tokens = [WordNetLemmatizer().lemmatize(word) for word in words]
    return tokens

def build_model():
    """function for building a model for classification of messages
    Input: none
    Output: model"""
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
                  
    parameters = {
        'vect__min_df':[1,5],
        'tfidf__use_idf':[True, False],
        'clf__estimator__n_estimators': [10,25],
        'clf__estimator__min_samples_split':[2,5] 
    }
    model  = GridSearchCV(pipeline, param_grid=parameters, cv=2) 
    return model 
                  
def evaluate_model(model, X_test, Y_test, category_names):
    """function for evaluating model against test-set
    Input: trained model, test features, test targets, categories
    Output: classification report showing precision, recall, f1_score and support metrics """
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names, digits=2))

def save_model(model, model_filepath):
    """function for saveing model to pickle file
    Input: trained model, filepath to pickle file """
    with open(model_filepath, 'wb') as pkl_file:
                  pickle.dump(model, pkl_file)
    pkl_file.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()