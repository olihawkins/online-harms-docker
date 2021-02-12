#=============================================================================#
#
#   This is an adaptation of the Online-Harms tool for use as a standalone tool
#   within a Docker image. This version currently supports Davidson and 
#   Wylcyzn models. Minor changes to the code have been introduced to support
#   csv rather than tsv, and for better compliance with PEP 8. The original 
#   authorship credit is preserved below.
# 
#   Original Author
#   Name:           Alexandros Mittos
#   Institute:      University College London
#   Initiated On:   March 6, 2020
#
#   Modified by
#   Name:           Oliver Hawkins
#   Institute:      University of Surrey (Visiting Fellow)
#   Initiated On:   February 5, 2021
#=============================================================================#

#===========================#
#        Imports            #
#===========================#

import datetime
import nltk
import numpy as np
import os
import pandas as pd
import subprocess
import sys
import time
nltk.download('all')


#===========================#
#        Variables          #
#===========================#

CLASSIFIERS = ['davidson', 'wulcyzn']

DATASETS = [
    'davidson', 
    'founta', 
    'gilbert', 
    'jinggab', 
    'jingreddit', 
    'kaggle', 
    'wazeem', 
    'wulcyzn', 
    'zampieri']

PREDICTION_MODE_BINARY = 'binary'
PREDICTION_MODE_PROBABILITY = 'probability'

ALL_RESULTS_PATH = 'results/'
pd.set_option('display.max_rows', 500)

#===========================#
#        Functions          #
#===========================#

def generate_reports(file, prediction_mode):
    """
    Generates a report for each trained model.
    """

    print('-----> Generating reports based on the trained models...')

    file_name = str(file).split('.')[0] # Get file name
    
    for classifier in CLASSIFIERS:
        
        # Get classifier path
        classifier_path = 'classifiers/' + \
            str(classifier) + '/' + \
            str(classifier) + \
            str('_classifier.py') 
        
        for dataset in DATASETS:
            print('-------> Generating report for classifier `' + \
                str(classifier) + '` trained on dataset `' + \
                str(dataset) + '`')
            
            # Get model path
            model_path = 'classifiers/' + classifier \
                + '/models/' + \
                classifier + '_' + \
                dataset + '.model' 

            # Get results path            
            results_path = 'results/' + \
                classifier + '_' + \
                dataset + '_' + \
                file_name + '_' + \
                'report.csv'

            # Rum the classifier against the data
            subprocess.call([
                'python',
                # Set the path of the classifier
                classifier_path,                     
                # Set the name of the classifier           
                '--classifier-name', str(classifier),
                # Path to the model
                '--model-path', model_path,
                # Set the process (train or test)
                '--test-path', str('dataset/') + file,
                # Set the path of the export model
                '--export-results-path', results_path,
                # Sets the prediction mode of the classifier
                '--prediction-mode', prediction_mode,
            ])


def generate_voting_ensemble(file, prediction_mode):
    """
    Simulates a voting ensemble based on all the results.
    """

    print('-----> Generating voting ensemble report...')

    file_name = str(file).split('.')[0]  # Get file name

    # Get all files in the directory
    results = [f for f in os.listdir(ALL_RESULTS_PATH) if not f.startswith('.')]

    # Combine all results into one dataframe
    list_of_frames = []
    for filename in results:
        df = pd.read_csv(ALL_RESULTS_PATH + filename, index_col=None, header=0)
        list_of_frames.append(df.iloc[:, 1:2]) # Get results column
    
    all_results_df = pd.concat(list_of_frames, axis=1, ignore_index=True)
    all_results_df['text'] = pd.read_csv(ALL_RESULTS_PATH + results[0])['text']
    
    if prediction_mode == PREDICTION_MODE_BINARY:

        all_results_df['label'] = all_results_df.apply(
            lambda row: int(row.mode()[0]), axis=1)
    
    elif prediction_mode == PREDICTION_MODE_PROBABILITY:
        
        numeric_results_df = all_results_df.select_dtypes(include=[np.number])
        all_results_df['label'] = numeric_results_df.apply(
            lambda row: row.mean(), axis=1) 

    ensemble_df = all_results_df[['text', 'label']]
    
    ensemble_df.to_csv(
        ALL_RESULTS_PATH + 'ensemble_report' + '_' + file_name + '.csv', 
        index=False)


#===========================#
#           Main            #
#===========================#

if __name__ == "__main__":

    # Set prediction mode
    prediction_mode = PREDICTION_MODE_BINARY

    if len(sys.argv) > 1 and sys.argv[1] == '-p':
        prediction_mode = PREDICTION_MODE_PROBABILITY

    # Start timer
    start = datetime.datetime.now()

    # Reading
    print('\nScanning for a .csv file in folder `dataset`...')

    files = [f for f in os.listdir('dataset') if not f.startswith('.')]

    for file in files:
        print('===> File to be labelled: ' + str(file))
        generate_reports(file, prediction_mode)
        generate_voting_ensemble(file, prediction_mode)

    print('Reporting complete: Results can be found in folder `results`')

    # End timer
    end = datetime.datetime.now()

    # Print results
    print("\nTotal time: " + str(end - start))

#===========================#
#       End of Script       #
#===========================#