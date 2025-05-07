import pandas as pd
import numpy as np
import importlib
import random, pickle
import neurokit2 as nk
import matplotlib.pyplot as plt
import sys, os, math, time
import glob as glob
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import eda_utils

import logging
from logging.config import dictConfig
import json
with open('logging.json', 'r') as read_file:
    contents = json.load(read_file)
dictConfig(contents)
LOGGER = logging.getLogger()

# Ignore all warnings
warnings.filterwarnings("ignore")

# Ignore warnings of a specific category
warnings.filterwarnings("ignore", category=FutureWarning)


def compute_features(window_data):
    return {
        'std_dev': np.std(window_data),
        'mean': np.mean(window_data),
        'min_value': np.min(window_data),
        'max_value': np.max(window_data)
    }

def extract_features(df_patient, sampling_freq, window_size):
    '''
    window_size in seconds.
    If freq is 4Hz, and the window_size is 10 secs, that's 40 samples
    '''
    window_length = window_size * sampling_freq
    # normalize raw EDA per patient
    df_patient['eda_signal'] = (
        df_patient['eda_signal'] - df_patient['eda_signal'].mean()
    ) / df_patient['eda_signal'].std()
    feature_vectors = []
    labels = []
    # Iterate through the data with the sliding window
    # if 'eda_signal_new_original' in df_patient:
    #     key = 'eda_signal_new_original'
    # else:
    #     key = 'eda_signal'
    key = 'eda_signal'

    for i in range(0, len(df_patient[key]), window_length):
        end_index = i + window_length
        window_data = df_patient[key].iloc[i:end_index]
        lbl_data = df_patient['label'].iloc[i:end_index]

        # Check if the window has sufficient data
        if len(window_data) == window_length:
            # Compute statistical features for the window
            features = compute_features(window_data)
            label = lbl_data.mode().iloc[0]
            # Append features to the feature vector list
            feature_vectors.append(features)
            labels.append(label)
    # Create a new DataFrame for the feature dataset
    feature_dataset = pd.DataFrame(feature_vectors)
    feature_dataset['label'] = labels
    LOGGER.debug("Window labels distribution: %s", pd.Series(labels).value_counts().to_dict())
    return feature_dataset

def generate_features(data_name, data_folder, freq, window_size):
    if data_name in ('WESAD', 'WESAD_chest'):
        classification_ids = pd.read_csv(f'data/WESAD_classification_ids.csv')
        ids = classification_ids['ID'].tolist() #['S10', 'S11']
        for patient_id in ids:
            LOGGER.debug(f'Extracting features for Patient {patient_id}')
            df = pd.read_csv(f'{data_folder}/{patient_id}/{patient_id}_eda.csv')
            features_df = extract_features(df, freq, window_size)
            features_df.to_csv(f'{data_folder}/{patient_id}/{patient_id}_eda_features.csv', index=False)
            LOGGER.debug(len(features_df))
    elif(data_name in ('sim_WESAD', 'sim_aug_WESAD', 'sim_WESAD_chest', 'sim_aug_WESAD_chest')):
        ids = [os.path.splitext(file)[0].split('_')[0] for file in os.listdir(data_folder) if file.endswith('_eda.csv')]
        for patient_id in ids:
            LOGGER.debug(f'Extracting features for Patient {patient_id}')
            df = pd.read_csv(f'{data_folder}/{patient_id}_eda.csv')
            features_df = extract_features(df, freq, window_size)
            features_df.to_csv(f'{data_folder}/{patient_id}_eda_features.csv', index=False)
            LOGGER.debug(len(features_df))
    LOGGER.debug("Done generating features")

def load_data(data_name, data_folder, ids):
    all_features = pd.DataFrame()
    if (data_name in ('WESAD', 'WESAD_chest')):
        for id in ids:
            features = pd.read_csv(f'{data_folder}/{id}/{id}_eda_features.csv')
            all_features = pd.concat([all_features, features], ignore_index=True)
    elif(data_name in ('sim_WESAD', 'sim_aug_WESAD', 'sim_WESAD_chest', 'sim_aug_WESAD_chest')):
        for id in ids:
            features = pd.read_csv(f'{data_folder}/{id}_eda_features.csv')
            all_features = pd.concat([all_features, features], ignore_index=True)
    return all_features


def model_training(data_name, mdl, train_data, test_data):
    #extract only labels 1 and 2 for real data ------0 1 for sim data
    LOGGER.debug(f'train_data.head(5)={train_data.head(5)}')
    # if data_name in ("WESAD", "WESAD_chest"):
    # Use labels as-is; just filter to keep valid binary labels
    # Keep only valid binary labels
    # if data_name.startswith("sim_"):
    #     train_data['label'] = train_data['label'].map({1:0, 2:1})
    #     test_data ['label']  = test_data ['label'].map({1:0, 2:1})

    train_data = train_data[train_data['label'].isin([0,1])]
    test_data  = test_data[test_data['label'].isin([0,1])]
    # Log label distributions for debugging
    LOGGER.debug("Train label distribution: %s", train_data['label'].value_counts().to_dict())
    LOGGER.debug("Test  label distribution: %s", test_data ['label'].value_counts().to_dict())

    # Ensure both classes are present
    if train_data['label'].nunique() < 2:
        LOGGER.warning("Only one class present in training data for fold: %s", train_data['label'].unique())
        # Return NaNs to skip this fold
        return {
            'Accuracy':       np.nan,
            'Precision':      np.nan,
            'Recall':         np.nan,
            'F1_Score':       np.nan,
            'True_Negative':  np.nan,
            'False_Positive': np.nan,
            'False_Negative': np.nan,
            'True_Positive':  np.nan
        }
    # else:
    #     train_data = train_data[(train_data['label'] == 1) | (train_data['label'] == 2)]
    #     test_data = test_data[(test_data['label'] == 1) | (test_data['label'] == 2)]

    # X_train = train_data.iloc[:, 1:-1]
    X_train = train_data.drop(columns=['label'])
    feature_cols = X_train.columns.tolist()
    LOGGER.debug('X_train.head(10):\n%s', X_train.head(10))

    y_train = train_data['label']
    LOGGER.debug('y_train.head(10):\n%s', y_train.head(10))

    # X_test = test_data.iloc[:, 1:-1]
    X_test = test_data.drop(columns=['label'])
    LOGGER.debug('X_test.head(10:\n%s', X_test.head(10))

    # y_test = test_data.iloc[:, -1]
    y_test = test_data['label']
    LOGGER.debug('y_test.head(10:\n%s', y_test.head(10))

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    if (mdl == 'LR'):
        model = LogisticRegression(class_weight='balanced')
    elif mdl == 'SVM':
        model = SVC(kernel='rbf', class_weight='balanced')
    elif mdl == 'RF':
        model = RandomForestClassifier(n_estimators=50, min_samples_split=5, class_weight='balanced')
    elif mdl == 'KNN':
        model = KNeighborsClassifier(n_neighbors=100)

    from sklearn.impute import SimpleImputer
    # 1) fit the imputer on your train matrix
    imp = SimpleImputer(strategy='mean')
    X_train_imputed = imp.fit_transform(X_train_scaled)
    X_test_imputed  = imp.transform(X_test_scaled)

    # 2) (optional) wrap back into a DataFrame for logging/debugging
    X_train = pd.DataFrame(X_train_imputed, columns=feature_cols)
    X_test  = pd.DataFrame(X_test_imputed,  columns=feature_cols)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)

    return {
        'Accuracy':accuracy_score(y_test, y_pred),
        'Precision':precision_score(y_test, y_pred),
        'Recall':recall_score(y_test, y_pred),
        'F1_Score':f1_score(y_test, y_pred),
        'True_Negative':conf_matrix[0,0],
        'False_Positive':conf_matrix[0,1],
        'False_Negative':conf_matrix[1,0],
        'True_Positive':conf_matrix[1,1]
    }


def main():
    data_name = str(sys.argv[1])
    data_folder = str(sys.argv[2])  #folder containing the patient files
    result_folder = str(sys.argv[3]) #folder to save the results files.
    freq = 4 if 'wrist' in result_folder.lower() else 12
    # window_size = 5
    window_size = 10
    LOGGER.debug(data_folder)

    #generate features
    generate_features(data_name, data_folder, freq, window_size)  #dont run this if you have already generated the features

    mdls = ['LR', 'SVM', 'RF', 'KNN'] #['LR'] #

    #classification
    if (data_name in ('WESAD', 'WESAD_chest')):
        ids = pd.read_csv(f'data/WESAD_classification_ids.csv')['ID'].tolist()
    elif(data_name in ('sim_WESAD', 'sim_aug_WESAD', 'sim_WESAD_chest', 'sim_aug_WESAD_chest')):
        ids = [os.path.splitext(file)[0].split('_')[0]  for file in os.listdir(data_folder) if file.endswith('_eda.csv')]
    LOGGER.debug(ids)
    for mdl in mdls:
        eval_lst = []
        pt_lst = []
        LOGGER.debug(f"Working on {mdl} model")
        for id in ids:
            LOGGER.debug(f"Testing patient {id}")
            train_ids = [id_ for id_ in ids if id_ != id]
            test_id = [id]
            #LOGGER.debug(f'{train_ids} | {test_id}')

            train_data = load_data(data_name, data_folder, train_ids)
            test_data = load_data(data_name, data_folder, test_id)

            train_data_shuffled = train_data.sample(frac=1).reset_index(drop=True)
            test_data_shuffled = test_data.sample(frac=1).reset_index(drop=True)

            eval = model_training(data_name, mdl, train_data_shuffled, test_data_shuffled)
            eval_lst.append(eval)
            pt_lst.append(id)
        results_df = pd.DataFrame(eval_lst)
        results_df['Patient_ID'] = pt_lst
        results_df = results_df[['Patient_ID', 'Accuracy', 'Precision', 'Recall', 'F1_Score','True_Negative',
                                    'False_Positive','False_Negative','True_Positive']]
        os.makedirs(result_folder, exist_ok=True)
        results_df.to_csv(f'{result_folder}/{mdl}.csv', index=False)
        LOGGER.debug('Classification Complete!')




if __name__ == "__main__":
    start_time = time.time()
    main()
    LOGGER.debug("--- %s seconds ---" % (time.time() - start_time))