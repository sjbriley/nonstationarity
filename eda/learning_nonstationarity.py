import pandas as pd
import numpy as np
import sys, os, math, time
import glob as glob
import pickle
import eda_utils
from numpy import mean
from collections import Counter
from xgboost import XGBClassifier
import logging
from datetime import datetime
from time import gmtime, strftime
from sklearn import metrics
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split

sys.path.append('..')
#import cpd_utils
import warnings

warnings.filterwarnings("ignore")

from logging.config import dictConfig
import json
with open('logging.json', 'r') as read_file:
    contents = json.load(read_file)
dictConfig(contents)
LOGGER = logging.getLogger()


def check_for_significance(pt_sample):
    remove_ckp_index = []
    bkps = pt_sample[pt_sample['cpd'] == 1].index #list of the breakpoints index
    for i in range(len(bkps)):
        #LOGGER.debug(f'-------------------------------bkp: {i}: index: {bkps[i]}')
        if(i == 0):
            pre_bkp = pt_sample[0:bkps[i]]
            post_bkp = pt_sample[bkps[i]:bkps[i+1]]
        elif(i == len(bkps)-1):
            pre_bkp = pt_sample[bkps[i-1]:bkps[i]]
            post_bkp = pt_sample[bkps[i]:len(pt_sample)]
        else:
            pre_bkp = pt_sample[bkps[i-1]:bkps[i]]
            post_bkp = pt_sample[bkps[i]:bkps[i+1]]

        sig_diff = eda_utils.check_sig_diff(pre_bkp['eda_signal'],post_bkp['eda_signal'])
        #LOGGER.debug(f'{pre_mean}:{post_mean}:{sig_diff}')
        if (sig_diff == False):
            remove_ckp_index.append(bkps[i])

    for i in remove_ckp_index:
        pt_sample.at[i,'cpd']=0

    bkps_after = pt_sample[pt_sample['cpd'] == 1].index

    LOGGER.debug(f"Number of bkps Before: {len(bkps)}: After:{len(bkps_after)}. Percentage dropped: {((len(bkps)-len(bkps_after))/len(bkps))*100}")
    LOGGER.debug(f"Class label Distribution \n {pt_sample['cpd'].value_counts()} \n {pt_sample['cpd'].value_counts(normalize=True)}")
    return pt_sample, len(bkps), len(bkps_after)

def type_of_change(pt_sample):
    means_list = [] #change in means list
    std_list = []
    type_of_change = []
    bkps = pt_sample[pt_sample['cpd'] == 1].index #list of the breakpoints index
    pt_sample['mean_change'] = np.nan
    pt_sample['std_change'] = np.nan
    pt_sample['type_of_change'] = np.nan
    for i in range(len(bkps)):
        #LOGGER.debug(f'-------------------------------bkp: {i}: index: {bkps[i]}')
        if(i == 0):
            pre_bkp = pt_sample[0:bkps[i]]
            post_bkp = pt_sample[bkps[i]:bkps[i+1]]
        elif(i == len(bkps)-1):
            pre_bkp = pt_sample[bkps[i-1]+1:bkps[i]]
            post_bkp = pt_sample[bkps[i]:len(pt_sample)]
        else:
            pre_bkp = pt_sample[bkps[i-1]:bkps[i]]
            post_bkp = pt_sample[bkps[i]:bkps[i+1]]
        pre_mean = mean(pre_bkp['eda_signal'])
        post_mean = mean(post_bkp['eda_signal'])

        diff_mean = post_mean - pre_mean
        diff_std = np.std(post_bkp['eda_signal']) -  np.std(pre_bkp['eda_signal'])
        #LOGGER.debug(f'Diff Mean={diff_mean}: Diff Std={diff_std}')
        #LOGGER.debug(f'Diff Mean={diff_mean}: Diff Std={diff_std}')

        #if (((diff_mean > 0.5) or (diff_mean < -0.5)) and ((diff_std > 0.5) or (diff_std < -0.5))):
        if (((diff_mean > 0) or (diff_mean < 0)) and ((diff_std > 0) or (diff_std < 0))):
            #LOGGER.debug("A change in both")
            type_of_change.append(3) #it's a change in both
            means_list.append(diff_mean)
            std_list.append(diff_std)
            pt_sample.at[bkps[i],'type_of_change']='mean+std'
            pt_sample.at[bkps[i],'mean_change']= diff_mean
            pt_sample.at[bkps[i],'std_change']= diff_std
        elif ((diff_mean > 0) or (diff_mean < 0)):
            #LOGGER.debug("A change in mean")
            type_of_change.append(1) # it's a change in mean
            means_list.append(diff_mean)
            pt_sample.at[bkps[i],'type_of_change']='mean'
            pt_sample.at[bkps[i],'mean_change']= diff_mean
        elif ((diff_std > 0) or (diff_std < 0)):
            #LOGGER.debug("A change in std")
            type_of_change.append(2)
            std_list.append(diff_std)
            pt_sample.at[bkps[i],'type_of_change']='std'
            pt_sample.at[bkps[i],'std_change']= diff_std

    return pt_sample,type_of_change, means_list, std_list


def calculate_metrics(y_actual, y_pred, y_prob):
    precision = precision_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, y_pred)
    f1 = f1_score(y_actual, y_pred)
    cm = confusion_matrix(y_actual, y_pred)

    auroc = roc_auc_score(y_actual, y_prob)
    precision, recall, thresholds = precision_recall_curve(y_actual, y_prob)
    auprc = auc(recall, precision)
    return precision, recall, accuracy, f1, cm, auroc, auprc



def build_cpd_model(X_train):
    LOGGER.debug("In building classifier method")
    ids = X_train['PID'].unique()
    LOGGER.debug('X_train.head(10) = %s', X_train.head(10))
    counter = Counter(X_train['y'])
    LOGGER.debug(f"count of 0: {counter[0]}, count of 1: {counter[1]}")
    # estimate scale_pos_weight value
    estimate = counter[0] / counter[1]
    LOGGER.debug('Estimate: %.3f' % estimate)
    accuracy = []
    f1score = []
    precision = []
    recall = []
    auroc_lst =[]
    auprc_lst = []
    model = XGBClassifier(scale_pos_weight=estimate)
    for id in ids:
        test_data = X_train[X_train['PID'] == id]
        train_data = X_train[X_train['PID'] != id]

        X_train_split = train_data.drop(columns=['y', 'PID'])
        y_train = train_data['y']
        X_test = test_data.drop(columns=['y', 'PID'])
        y_test = test_data['y']

        LOGGER.debug(f"0s: {pd.Series(y_test).value_counts()[0]}, 1s: {pd.Series(y_test).value_counts()[1]}")
            # Perform stratified undersampling on the training set
        #X_train_undersampled, y_train_undersampled = stratified_undersampling(X_train_split, y_train)
        #LOGGER.debug(f"X_train before undersampling: {len(X_train_split)}. After: {len(X_train_undersampled)}")
        model.fit(X_train_split, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        LOGGER.debug(f"Predicted prob: {y_prob}")
        #metrics_report = classification_report(y_test, y_pred, output_dict=True)
        precision_, recall_, acc, f1, cm, auroc, auprc = calculate_metrics(y_test, y_pred, y_prob)
        out_dir = "models/classifier_predictions/"
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame({'y_actual': y_test, 'y_pred': y_pred, 'y_prob': y_prob}).to_csv(f"{out_dir}/{id}.csv", index=False)
        LOGGER.debug(f"ID: {id}. p|r|acc|f1|cm: {precision_, recall_, acc, f1, cm.ravel()}") #TN, FP,FN,TP
        LOGGER.debug(f"AUROC: {auroc} | AUPRC: {auprc}")
        conf = metrics.confusion_matrix(y_test, y_pred).ravel()
        LOGGER.debug(f"XGB Conf matrix: {conf}") #tn, fp, fn, tp
        LOGGER.debug(f"XGB Conf matrix: {conf}") #tn, fp, fn, tp
        accuracy.append(acc)
        f1score.append(f1)
        precision.append(precision_)
        recall.append(recall_)
        auroc_lst.append(auroc)
        auprc_lst.append(auprc)

    mdl_result = pd.DataFrame({
                'PID': ids,
                'Accuracy': accuracy,
                'F1score':f1score,
                'Precision':precision,
                'Recall':recall,
                'AUROC':auroc_lst,
                'AUPRC':auprc_lst
            })
    os.makedirs(f"models/WESAD", exist_ok=True)
    mdl_result.to_csv(f"models/WESAD/classifier_021024.csv", index=False)
    X_train_total = X_train.drop(columns=['y', 'PID'])
    y_train_total = X_train['y']
    model.fit(X_train_total, y_train_total)
    return model


def get_change_type_dist(all_type):
    '''
    to generate the probabilities of each type of change
    all_type: the list of all the types of change for each changepoint
    '''
    mappings = {1:"mean", 2:"std", 3:"mean+std"}
    df_type = pd.DataFrame({'freq': all_type})
    df_type = df_type.groupby('freq', as_index=False).size()
    df_type = df_type.rename(columns={"freq": "type_of_change_key", "size": "freq"})
    df_type["type"] = df_type["type_of_change_key"].replace(mappings)
    #find the probabilities
    df_type['prob'] = df_type['freq']/sum(df_type['freq'])
    #save it into dictionary
    type_of_change_list = df_type[['type', 'prob', 'freq']].to_dict('list')
    return type_of_change_list

def main():
    global DATASET
    DATASET = str(sys.argv[1]) #get the type of data
    input_folder = str(sys.argv[2]) #get the type of data

    all_type_of_change = []
    all_means = []
    all_std = []
    all_duration = []
    all_eda_diff = []
    bkps_before = [] #number of bkps before checking for significance
    bkps_after = []
    pt_list = []
    y_train_total = []

    learning_Ids = pd.read_csv(f'data/{DATASET}_learning_ids.csv')['ID'].tolist() #['S4', 'S9']
    annonated_files = str(sys.argv[3]) #folder ro save them in
    frequency = 4 #700Hz
    window_size =  int(str(sys.argv[4]))
    bkps_count_file = str(sys.argv[5])

    mdl_file = str(sys.argv[6]) #pickle file

    LOGGER.debug('learning_ids: %s, annotated_files: %s, frequency: %s, window_size: %s, bkps_count_file: %s, mdl_file: %s',
                 learning_Ids, annonated_files, frequency, window_size, bkps_count_file, mdl_file)

    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    LOGGER.debug(f"{now}-----------------------------NOW STARTING A NEW RUN- LEARNING_NOSNSTATIONARITY.PY")

    LOGGER.debug(f"Number of patients: {len(learning_Ids)}")

    all_pt = pd.DataFrame()
    X_train_total = pd.DataFrame()

    #------------------------------------------comment just to run classifier
    for i in range(len(learning_Ids)):
        patient_id = learning_Ids[i]
        LOGGER.debug(f"i: {i} || Working on patient: {learning_Ids[i]}")

        #load eda data
        pt_sample = pd.read_csv(f'{input_folder}/{patient_id}/{patient_id}_eda.csv')
        pt_sample['cpd'] = 0
        LOGGER.debug(f"i: {i} || Working on patient: {learning_Ids[i]}. Patient length: {len(pt_sample)}")
        pt_sample = pt_sample.dropna()
        #pt_sample.set_index("dates", inplace=True)

        LOGGER.debug("Get the change point and duration")
        trends = eda_utils.identify_df_trends(pt_sample, "eda_signal", window_size)
        LOGGER.debug('Trends: %s', trends)
        #result, trends = cpd_utils.large_missing_intervals(trends, 120) #no need since there is no timestamp
        pt_data = eda_utils.annotate_cp_duration(pt_sample, trends) #annotate the data with the duration and cp
        LOGGER.debug('pt_data (annotated with duration and cp from trends): %s', pt_data)

        #drop the non-significant
        pt_data, len_bkps, len_bkps_after = check_for_significance(pt_data)
        bkps_before.append(len_bkps) #number of bkps before checking for significance
        bkps_after.append(len_bkps_after)
        LOGGER.debug(f"Number of bkps before checking for significance: {len_bkps}. After checking: {len_bkps_after}")
        pt_list.append(learning_Ids[i])

        #get the duration from the samples length
        pt_data['duration'] = pt_data['duration_samples']/frequency #in seconds
        LOGGER.debug('pt_data["duration"] (after dividing duration_samples by frequency) is:\n%s', pt_data['duration'])

        #find the drastic jumps in the data (glucose at t - glucose at t-1)
        #pt_data['time_diff'] = pt_data['dates'].diff()
        pt_data['diff'] = pt_data['eda_signal'].diff()
        eda_diff = pt_data['diff']
        eda_diff_list = eda_diff.tolist()

        if (len_bkps_after > 1):
            LOGGER.debug('More than breakpoint found: %s', len_bkps_after)
            #get the duration
            duration = pt_data[pt_data['cpd'] == 1]['duration']
            duration_list = duration.tolist()

            #get type and intensity of change
            LOGGER.debug("Getting the change type and intensity")
            pt_data, change_type, means, std = type_of_change(pt_data)
            LOGGER.debug(f"Length of the change types, means, std: {len(change_type)} {len(means)} {len(std)}")

            #append the type and intensity of change
            all_type_of_change.extend(change_type)
            all_means.extend(means)
            all_std.extend(std)
            all_duration.extend(duration_list)
            all_eda_diff.extend(eda_diff_list)

            ## save the files
            file_name = f"{learning_Ids[i]}.csv"
            file_path = os.path.join(annonated_files,file_name)
            LOGGER.debug('saving pt_data to %s', file_path)
            os.makedirs(annonated_files, exist_ok=True)
            pt_data.to_csv(file_path, index=False)

            #append the patient into a big df
            #LOGGER.debug(pt_data.head(5))
            all_pt = pd.concat([all_pt, pt_data], axis=0).reset_index(drop=True)
            LOGGER.debug('all_pt = pd.concat([all_pt, pt_data], axis=0).reset_index(drop=True); all_pt.tail(t5) = %s', all_pt.tail(5))

            #get features
            LOGGER.debug(f"Number of rows in pt data: {len(pt_data)}")
            X_train, y_train, data = eda_utils.get_features(pt_data, DATASET, frequency, window_size)
            zeros = Counter(y_train)[0]
            ones = Counter(y_train)[1]
            LOGGER.debug(f"Number of rows in X_train data: {len(X_train)}. in y_train: {len(y_train)}: len of 0={zeros}, 1={ones}")

            ## save the features to verify correctness
            file_name = f"{learning_Ids[i]}_features.csv"
            file_path = os.path.join(annonated_files,file_name)
            LOGGER.debug('saving X_train features to %s', file_path)
            os.makedirs(annonated_files, exist_ok=True)
            X_train['PID'] = learning_Ids[i]
            X_train.to_csv(file_path, index=False)

            #y_train = data['cpd']
            X_train_total = pd.concat([X_train_total, X_train], axis=0).reset_index(drop=True)
            y_train_total.extend(y_train)
        else:
            LOGGER.debug("Only one breakpoint found")


    #distribution of the type of change
    LOGGER.debug(f"Full length of all: {len(all_pt)}")
    LOGGER.debug(f"Full length. || X_train data: {len(X_train_total)}. in y_train: {len(y_train_total)}")
    type_of_change_list = get_change_type_dist(all_type_of_change)

    #build classifier model for change point detection
    X_train_total['y'] = y_train_total #combine into one df
    os.makedirs("models", exist_ok=True)
    X_train_total.to_csv(f"models/{DATASET}_classifier_data.csv", index=False)
    LOGGER.debug('all_pt.tail(5) = %s', all_pt.tail(5))
    #'''#--------------------------------------------------------------------comment just to run classifier
    # X_train_total = pd.read_csv(f"models/{DATASET}_classifier_data.csv")
    LOGGER.debug(f"Total rows for all patients: {len(X_train_total)}")
    cpd_mdl = build_cpd_model(X_train_total)
    LOGGER.debug("Training completed")

    #save the distribution to pkl
    mdl = {"mdl":cpd_mdl, "type_of_change":type_of_change_list,
    "mean_change": all_means, "std_change": all_std, "duration":all_duration, "eda_diff":all_eda_diff}

    # save dictionary to pkl file
    LOGGER.debug('saving mdl to %s', mdl_file)
    eda_utils.save_to_pkl(mdl_file, mdl)

    #save the number of changepoints before and after checking for sig
    bkps_df = pd.DataFrame()
    bkps_df['PtID'] = pt_list
    bkps_df['before_checking'] = bkps_before
    bkps_df['after_checking'] = bkps_after

    bkps_df.to_csv(bkps_count_file, index=False)
    LOGGER.debug('saving %s to %s', bkps_df, bkps_count_file)
    LOGGER.debug(f"{now}-----------------------------END OF THE RUN- LEARNING.PY")


if __name__ == "__main__":
    start_time = time.time()
    main()
    LOGGER.debug("--- %s seconds ---" % (time.time() - start_time))
