# Disaster-Response-Pipeline
Analyze real disaster data from Figure Eight to build a model for an API that classifies disaster messages.

The project includes a machine learning pipeline to categorize these events so that the messages can be sent to an appropriate disaster relief agency, a web app where an emergency worker can input a new message and get classification results in several categories. The web app also display visualizations of message data. 
The project is a part of a Udacity Data Scientist Nanodegree.

Templates for the scripts were provided in the Udacity workspace. The programs/scripts are written in python.

## Project Components
The project consists of three components:

## 1. ETL Pipeline functions
•	Load messages- and categories datasets

•	Merge the two datasets

•	Clean the dataset

•	Store the cleaned dataset in a SQLite database

## 2. ML Pipeline functions
•	Load dataset from the SQLite database

•	Split the dataset into training and test sets

•	Build a text processing and machine learning pipeline

•	Train and tunes a model using GridSearchCV

•	Output results on the test set using classification-report

Export the final model as a pickle file

## 3. Flask web app
Using flask, html, css and javascript to:

•	Categorize new messages

•	Visualize of the training set genres, categories and news media categories


## How to run the program
•	In the data directory - run ETL pipeline that cleans data and stores in database:  python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

•	In the model directory - run ML pipeline that trains classifier and save models: python train_classifier.py ../data/DisasterResponse.db.classifier.pkl

•	In the app directory - run the web app - python run.py

Go to http://0.0.0.0:3001/

## File structure (given by the project spesification from Udacity):
File structure of the project is:
- app
|- template
|- master.html  #main page of web app
|- go.html  #classification result page of web app
|- run.py  #Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- Notebook
| - ETL Pipeline Preparation
| - ML Pipeline Preparation

- README.md


 

 
