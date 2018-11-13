import pandas as pd
import numpy as np

import sys
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
   """ function for load message and categories files, merge and store them in a dataframe
   Input: filepaths for messages.csv and categories.csv data
   Output: merged dataframe """
   messages = pd.read_csv(messages_filepath, delimiter=",")
   categories = pd.read_csv(categories_filepath,  delimiter=",")
   df = messages.merge(categories, on= 'id')
   return df

def clean_data(df):
    """ function for cleaning a dataframe 
    Input: dataframe
    Output: dataframe with messages, categories as column names and removed duplicates """
    categories = df['categories'].str.split(";", expand=True)
    row = categories.iloc[0]
    categories.columns = row.apply(lambda x:x.split('-')[0])
    
    for column in categories:
    #set value to be the last character of the string
        categories[column] = categories[column] = categories[column].str.split('-').str[-1]
    #convert from string to int
        categories[column] = categories[column].astype(int)
    #assert that all columns have values 0 or 1
        categories[column] = categories[column].apply(lambda x:1 if x>1 else x)
    #drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    #concatenate original dataframe with new 'categories' dataframe
    df = pd.concat([df, categories], axis=1)
    #drop duplicates
    df.drop_duplicates(inplace=True)
    return df
    
def save_data(df, database_filename):
    """ function for saving the dataframe to a SQLite database 
    Input: dataframe and database_filename: given input filename from user
    Output: database with saved data """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterData', engine, if_exists='replace', index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterData.db')


if __name__ == '__main__':
    main()