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
import subprocess
import os
import pandas as pd
import nltk
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

all_results_path = 'results/'
pd.set_option('display.max_rows', 500)

#===========================#
#        Functions          #
#===========================#

def generate_reports(file):
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
            ])


def generate_voting_ensemble(file):
    """
    Simulates a voting ensemble based on all the results.
    """

    print('-----> Generating voting ensemble report...')

    file_name = str(file).split('.')[0]  # Get file name

    # Get all files in the directory
    results = [f for f in os.listdir(all_results_path) if not f.startswith('.')]

    # Combine all results into one dataframe
    list_of_frames = []
    for filename in results:
        df = pd.read_csv(all_results_path + filename, index_col=None, header=0)
        list_of_frames.append(df.iloc[:, 1:2]) # Get results column
    
    all_results_df = pd.concat(list_of_frames, axis=1, ignore_index=True)
    all_results_df['text'] = pd.read_csv(all_results_path + results[0])['text']
    
    all_results_df['label'] = all_results_df.apply(
        lambda row: int(row.mode()[0]), axis=1)
    
    ensemble_df = all_results_df[['text', 'label']]
    
    ensemble_df.to_csv(
        all_results_path + 'ensemble_report' + file_name + '.csv', 
        index=False)


#===========================#
#           Main            #
#===========================#

if __name__ == "__main__":

    # Start timer
    start = datetime.datetime.now()

    # Reading
    print('\nScanning for a .csv file in folder `dataset`...')

    files = [f for f in os.listdir('dataset') if not f.startswith('.')]

    for file in files:
        print('===> File to be labelled: ' + str(file))
        generate_reports(file)
        generate_voting_ensemble(file)

    print('Reporting complete: Results can be found in folder `results`')

    # End timer
    end = datetime.datetime.now()

    # Print results
    print("\nTotal time: " + str(end - start))

#===========================#
#       End of Script       #
#===========================#