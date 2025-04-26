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
    feature_vectors = []
    labels = []
    # Iterate through the data with the sliding window
    for i in range(0, len(df_patient['eda_signal']), window_length):
        end_index = i + window_length
        window_data = df_patient['eda_signal'].iloc[i:end_index]
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
    return feature_dataset

def generate_features(data_name, data_folder, freq, window_size):
    if (data_name == 'WESAD'):
        classification_ids = pd.read_csv(f'data/{data_name}_classification_ids.csv')
        ids = classification_ids['ID'].tolist() #['S10', 'S11']
        for patient_id in ids:
            print(f'Extracting features for Patient {patient_id}')
            df = pd.read_csv(f'{data_folder}/{patient_id}/{patient_id}_eda.csv')
            features_df = extract_features(df, freq, window_size)
            features_df.to_csv(f'{data_folder}/{patient_id}/{patient_id}_eda_features.csv')
            print(len(features_df))
    elif(data_name == 'sim_WESAD' or  data_name == 'sim_aug_WESAD'):
        ids = [os.path.splitext(file)[0].split('_')[0] for file in os.listdir(data_folder) if file.endswith('_eda.csv')]
        for patient_id in ids:
            print(f'Extracting features for Patient {patient_id}')
            df = pd.read_csv(f'{data_folder}/{patient_id}_eda.csv')
            features_df = extract_features(df, freq, window_size)
            features_df.to_csv(f'{data_folder}/{patient_id}_eda_features.csv')
            print(len(features_df))
    print("Done generating features")

def load_data(data_name, data_folder, ids):
    all_features = pd.DataFrame()
    if (data_name == 'WESAD'):
        for id in ids:
            features = pd.read_csv(f'{data_folder}/{id}/{id}_eda_features.csv')
            all_features = all_features.append(features, ignore_index=True)
    elif(data_name == 'sim_WESAD' or data_name == 'sim_aug_WESAD'):
        for id in ids:
            features = pd.read_csv(f'{data_folder}/{id}_eda_features.csv')
            all_features = all_features.append(features, ignore_index=True)
    return all_features


def model_training(data_name, mdl, train_data, test_data):
    #extract only labels 1 and 2 for real data ------0 1 for sim data
    print(train_data.head(5))
    if (data_name=="WESAD"):
        train_data = train_data[(train_data['label'] == 1) | (train_data['label'] == 2)]
        test_data = test_data[(test_data['label'] == 1) | (test_data['label'] == 2)]
    else:
        train_data = train_data[(train_data['label'] == 0) | (train_data['label'] == 1)]
        test_data = test_data[(test_data['label'] == 0) | (test_data['label'] == 1)]
    X_train = train_data.iloc[:, 1:-1]
    print(X_train.head(10))
    y_train = train_data.iloc[:, -1]
    #print(y_train.head(10))
    X_test = test_data.iloc[:, 1:-1]
    #print(X_train.head(10))
    y_test = test_data.iloc[:, -1]
    if (mdl == 'LR'):
        model = LogisticRegression()
    elif mdl == 'SVM':
        model = SVC(kernel='rbf')
    elif mdl == 'RF':
        model = RandomForestClassifier(n_estimators=50, min_samples_split=5)
    elif mdl == 'KNN':
        model = KNeighborsClassifier(n_neighbors=100)

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
    freq = 4
    window_size = 10
    print(data_folder)

    #generate features
    generate_features(data_name, data_folder, freq, window_size)  #dont run this if you have already generated the features

    mdls = ['LR', 'SVM', 'RF', 'KNN'] #['LR'] #

    #classification
    if (data_name == 'WESAD'):
        ids = pd.read_csv(f'data/{data_name}_classification_ids.csv')['ID'].tolist()
    elif(data_name == 'sim_WESAD' or data_name == 'sim_aug_WESAD'):
        ids = [os.path.splitext(file)[0].split('_')[0]  for file in os.listdir(data_folder) if file.endswith('_eda.csv')]
    print(ids)
    for mdl in mdls:
        eval_lst = []
        pt_lst = []
        print(f"Working on {mdl} model")
        for id in ids:
            print(f"Testing patient {id}")
            train_ids = [id_ for id_ in ids if id_ != id]
            test_id = [id]
            #print(f'{train_ids} | {test_id}')

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
        results_df.to_csv(f'{result_folder}/{mdl}.csv')
        print('Classification Complete!')




if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))