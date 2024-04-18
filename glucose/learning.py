import pandas as pd
import numpy as np 
import sys, os, math, time
import glob as glob
import pickle
import cpd_utils
from numpy import mean
from collections import Counter
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import logging
from datetime import datetime
from time import gmtime, strftime
from sklearn import metrics
from datetime import datetime, timedelta
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc



def print_to_file(statement):
    file = f'debug_{DATASET}_learning.txt'

    logging.basicConfig(filename=file, level=logging.DEBUG, format='')
    
    logging.debug(statement)
    
def check_for_significance(pt_sample):
    remove_ckp_index = []
    bkps = pt_sample[pt_sample['cpd'] == 1].index #list of the breakpoints index 
    for i in range(len(bkps)):
        #print(f'-------------------------------bkp: {i}: index: {bkps[i]}')
        if(i == 0):
            pre_bkp = pt_sample[0:bkps[i]]
            post_bkp = pt_sample[bkps[i]:bkps[i+1]]
        elif(i == len(bkps)-1):
            pre_bkp = pt_sample[bkps[i-1]:bkps[i]]
            post_bkp = pt_sample[bkps[i]:len(pt_sample)]
        else:
            pre_bkp = pt_sample[bkps[i-1]:bkps[i]]
            post_bkp = pt_sample[bkps[i]:bkps[i+1]]

        sig_diff = cpd_utils.check_sig_diff(pre_bkp['glucose_level'],post_bkp['glucose_level'])
        #print(f'{pre_mean}:{post_mean}:{sig_diff}') 
        if (sig_diff == False):
            remove_ckp_index.append(bkps[i])
            
    for i in remove_ckp_index:
        pt_sample.at[i,'cpd']=0
        
    bkps_after = pt_sample[pt_sample['cpd'] == 1].index 

    print(f"Number of bkps Before: {len(bkps)}: After:{len(bkps_after)}. Percentage dropped: {((len(bkps)-len(bkps_after))/len(bkps))*100}")
    print(f"Class label Distribution \n {pt_sample['cpd'].value_counts()} \n {pt_sample['cpd'].value_counts(normalize=True)}")
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
        #print_to_file(f'-------------------------------bkp: {i}: index: {bkps[i]}')
        if(i == 0):
            pre_bkp = pt_sample[0:bkps[i]]
            post_bkp = pt_sample[bkps[i]:bkps[i+1]]
        elif(i == len(bkps)-1):
            pre_bkp = pt_sample[bkps[i-1]+1:bkps[i]]
            post_bkp = pt_sample[bkps[i]:len(pt_sample)]
        else:
            pre_bkp = pt_sample[bkps[i-1]:bkps[i]]
            post_bkp = pt_sample[bkps[i]:bkps[i+1]]
        pre_mean = mean(pre_bkp['glucose_level'])
        post_mean = mean(post_bkp['glucose_level'])
        
        diff_mean = post_mean - pre_mean
        diff_std = np.std(post_bkp['glucose_level']) -  np.std(pre_bkp['glucose_level']) 
        #print(f'Diff Mean={diff_mean}: Diff Std={diff_std}')
        #print_to_file(f'Diff Mean={diff_mean}: Diff Std={diff_std}')
        
        #if (((diff_mean > 0.5) or (diff_mean < -0.5)) and ((diff_std > 0.5) or (diff_std < -0.5))):
        if (((diff_mean > 0) or (diff_mean < 0)) and ((diff_std > 0) or (diff_std < 0))):
            #print_to_file("A change in both")
            type_of_change.append(3) #it's a change in both
            means_list.append(diff_mean)
            std_list.append(diff_std)
            pt_sample.at[bkps[i],'type_of_change']='mean+std'
            pt_sample.at[bkps[i],'mean_change']= diff_mean
            pt_sample.at[bkps[i],'std_change']= diff_std
        elif ((diff_mean > 0) or (diff_mean < 0)):
            #print_to_file("A change in mean")
            type_of_change.append(1) # it's a change in mean 
            means_list.append(diff_mean)
            pt_sample.at[bkps[i],'type_of_change']='mean'
            pt_sample.at[bkps[i],'mean_change']= diff_mean
        elif ((diff_std > 0) or (diff_std < 0)):
            #print_to_file("A change in std")
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



def build_cpd_model(X_train,dataset):
    print("In building classifier method")
    print(X_train.head(10))
    counter = Counter(X_train['y'])
    # estimate scale_pos_weight value
    estimate = counter[0] / counter[1]
    print_to_file(f"Imbalance class: 0s : {counter[0]}. 1: {counter[1]}")
    print('Estimate: %.3f' % estimate)
    class_weights = compute_class_weight('balanced', classes=[0, 1], y=X_train['y'])

    models = [XGBClassifier(scale_pos_weight=estimate)]
    #models = [XGBClassifier(scale_pos_weight='balanced')]
    max_f1 = 0
    #XGBClassifier(scale_pos_weight=estimate)
    ids = X_train['PID'].unique() #get the unique IDs

    if (dataset == 'ohio'):
        #LOOCV
        for mdl in models:
            model_name = mdl.__class__.__name__
            print(f"Model {model_name}")
            accuracy = []
            f1score = []
            precision = []
            recall = []
            auroc_lst =[]
            auprc_lst = []
            for id in ids:
                test_data = X_train[X_train['PID'] == id]
                train_data = X_train[X_train['PID'] != id]

                X_train_split = train_data.drop(columns=['y', 'PID', 'Unnamed: 0'])
                print(X_train_split.columns)
                y_train = train_data['y']
                X_test = test_data.drop(columns=['y', 'PID', 'Unnamed: 0'])
                y_test = test_data['y']

                print_to_file(f"0s: {pd.Series(y_test).value_counts()[0]}, 1s: {pd.Series(y_test).value_counts()[1]}")
                 # Perform stratified undersampling on the training set
                #X_train_undersampled, y_train_undersampled = stratified_undersampling(X_train_split, y_train)
                #print_to_file(f"X_train before undersampling: {len(X_train_split)}. After: {len(X_train_undersampled)}")
                mdl.fit(X_train_split, y_train)

                y_pred = mdl.predict(X_test)
                y_prob = mdl.predict_proba(X_test)[:,1]
                print_to_file(f"Predicted prob: {y_prob}")
                
                precision_, recall_, acc, f1, cm, auroc, auprc = calculate_metrics(y_test, y_pred, y_prob)
                #save the predictions....for evaluating diff thresholds
                pd.DataFrame({'y_actual': y_test, 'y_pred': y_pred, 'y_prob': y_prob}).to_csv(f"models/{dataset}_classifier_predictions/{id}.csv")
                print_to_file(f"ID: {id}. {model_name}: p|r|acc|f1|cm: {precision_, recall_, acc, f1, cm.ravel()}") #TN, FP,FN,TP
                print_to_file(f"AUROC: {auroc} | AUPRC: {auprc}")
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

            mdl_result.to_csv(f"models/{dataset}/classifier/{model_name}.csv")
            #avg acc and f1 score
            avg_acc = sum(accuracy) / len(accuracy)
            avg_f1 = sum(f1score) / len(f1score)

            #keep track of best model XGB clasisfier was the best...use that going forward.
            if(True):
                max_f1 = avg_f1
                best_mdl = mdl    
    elif dataset == 'oaps':
        accuracy = []
        f1score = []
        precision = []
        recall = []
        mdl_names = []
        auroc_lst = []
        auprc_lst = []
        # 80/20 split for training and testing
        train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=42)
        print_to_file(f"train ids- {train_ids}, test ids - {test_ids}")
        train_data = X_train[X_train['PID'].isin(train_ids)]
        X_train_split = train_data.drop(columns=['y', 'PID', 'Unnamed: 0'])
        y_train_split = train_data['y']

        test_data = X_train[X_train['PID'].isin(test_ids)]
        X_test = test_data.drop(columns=['y', 'PID', 'Unnamed: 0'])
        y_test = test_data['y']
        print_to_file(f"Test data: 0s: {pd.Series(y_test).value_counts()[0]}, 1s: {pd.Series(y_test).value_counts()[1]}")
        print_to_file(f"Train data: 0s: {pd.Series(y_train_split).value_counts()[0]}, 1s: {pd.Series(y_train_split).value_counts()[1]}")
        for mdl in models:
            model_name = mdl.__class__.__name__
            print(f"Model {model_name}")

             # Perform stratified undersampling on the training set
            X_train_undersampled, y_train_undersampled = stratified_undersampling(X_train_split, y_train_split)
            print_to_file(f"X_train before undersampling: {len(X_train_split)}. After: {len(X_train_undersampled)}")
            mdl.fit(X_train_split, y_train_split)

            y_pred = mdl.predict(X_test)
            y_prob = mdl.predict_proba(X_test)[:,1]
            precision_, recall_, acc, f1, cm, auroc, auprc = calculate_metrics(y_test, y_pred, y_prob)
            #save the predictions....for evaluating diff thresholds
            pd.DataFrame({'PID':test_data['PID'].tolist(),'y_actual': y_test, 'y_pred': y_pred, 'y_prob': y_prob}).to_csv(f"models/{dataset}_classifier_predictions.csv")
            print_to_file(f"{model_name}: p|r|acc|f1|cm: {precision_, recall_, acc, f1, cm.ravel()}") #TN, FP,FN,TP
            print_to_file(f"AUROC: {auroc} | AUPRC: {auprc}")

            # Append metrics to lists
            accuracy.append(acc)
            f1score.append(f1)
            precision.append(precision_)
            recall.append(recall_)
            mdl_names.append(model_name)
            auroc_lst.append(auroc)
            auprc_lst.append(auprc)

            #keep track of best model 
            if(True): #if(f1 > max_f1):
                max_f1 = f1
                best_mdl = mdl

        mdl_result = pd.DataFrame({
            'Model': mdl_names,
            'Accuracy': accuracy,
            'F1-score': f1score,
            'Precision': precision,
            'Recall': recall,
            'AUROC':auroc_lst,
            'AUPRC':auprc_lst
        })

        mdl_result.to_csv(f"models/{dataset}/classifier.csv", index=False)
    print_to_file(f"Best model: {best_mdl.__class__.__name__}. F1 score: {max_f1}")
    print(f"Best model: {best_mdl.__class__.__name__}. F1 score: {max_f1}")
    #use the best mdl to train on all to get the final mdl
    X_train_total = X_train.drop(columns=['y', 'PID'])
    y_train_total = X_train['y']
    best_mdl.fit(X_train_total, y_train_total)  
    return best_mdl




def stratified_undersampling(X_train, y_train):
    # Concatenate the features and target labels
    train_data = pd.concat([X_train, y_train], axis=1)

    # Separate majority and minority classes
    majority_class = train_data[train_data['y'] == 0]
    minority_class = train_data[train_data['y'] == 1]

    # Perform stratified undersampling on the majority class
    undersampled_majority = resample(majority_class,
                                     replace=False,  # sample without replacement
                                     n_samples=len(minority_class)*2,  # match minority class
                                     random_state=42)  # reproducible results

    # Combine minority class with undersampled majority class
    undersampled_data = pd.concat([undersampled_majority, minority_class])

    # Shuffle the data
    undersampled_data = undersampled_data.sample(frac=1, random_state=42)

    # Separate features and target labels again
    X_train_undersampled = undersampled_data.drop('y', axis=1)
    y_train_undersampled = undersampled_data['y']

    return X_train_undersampled, y_train_undersampled


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

def group_into_daily(patient):
    '''
    Function to group a patient's readings into days

    patient: the patient's csv file
    '''
    #extract it into df per day
    patient.dates = pd.to_datetime(patient.dates)
    patient_daily = []
    for i in patient.groupby(patient.dates.dt.floor('d')):
        patient_daily.append(i[1])
    print(f"There are {len(patient_daily)} days data in this file")
    return patient_daily

def main():
    print("In Main")
    global DATASET
    DATASET = str(sys.argv[1]) #get the type of data

    all_type_of_change = []
    all_means = []
    all_std = []
    all_duration = []
    
    all_glucose_diff = []

    bkps_before = [] #number of bkps before checking for significance
    bkps_after = []
    pt_list = []
    
    y_train_total = []

    if (DATASET=='oaps'):
        print (DATASET)
        input_df = str(sys.argv[2]) #csv file of the input data
        learning_Ids = str(sys.argv[3])  #csv file containing the learning IDS
        annonated_files = str(sys.argv[4]) # folder to save the subject files annotated with cpd"
        learning_Ids = learning_Ids['PID'].tolist()
        print(learning_Ids)
        df_new = input_df[input_df['Pt_ID'].isin(learning_Ids)]
        df_new.rename({'Pt_ID': 'PtID'}, axis=1, inplace=True)
    
    elif(DATASET=='ohio'):
        #TO DO include ohio data link here
        input_df = str(sys.argv[2])
        annonated_files = str(sys.argv[4])
        learning_Ids = str(sys.argv[3]) 
        df_new = input_df

    bkps_count_file = f'models/{DATASET}/{DATASET}_bkps_count.csv'
    mdl_file = f'models/{DATASET}/{DATASET}_mdl.pkl'

    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print_to_file(f"{now}-----------------------------NOW STARTING A NEW RUN- LEARNING.PY")
        
    #df_new['dates'] = pd.to_datetime(df_new['dates'])
    df_new2 = df_new[['PtID', 'dates', 'glucose_level']]
    df_new2['cpd'] = 0
    print(f"Number of pts: {len(learning_Ids)} || Length of data: {len(df_new2)}")
    print_to_file(f"Number of pts: {len(learning_Ids)} || Length of data: {len(df_new2)}")

    all_pt = pd.DataFrame()
    X_train_total = pd.DataFrame()

    #'''------------------------------------------comment just to run classifier
    for i in range(len(learning_Ids)): #len(learning_Ids)
        print(f"i: {i} || Working on patient: {learning_Ids[i]}")
        pt_sample = df_new2[df_new2['PtID'] == int(learning_Ids[i])]
        print(pt_sample.head())
        print_to_file(f"i: {i} || Working on patient: {learning_Ids[i]}. Patient length: {len(pt_sample)}")
        pt_sample = pt_sample.dropna()
        pt_sample.set_index("dates", inplace=True)
        
        print_to_file("Get the change point and duration")
        trends = cpd_utils.identify_df_trends(pt_sample, "glucose_level")
        result, trends = cpd_utils.large_missing_intervals(trends, 120) 
        pt_data = cpd_utils.annonate_cp_duration(trends) #annotate the data with the duration and cp

        #drop the non-significant
        pt_data, len_bkps, len_bkps_after = check_for_significance(pt_data)
        bkps_before.append(len_bkps) #number of bkps before checking for significance
        bkps_after.append(len_bkps_after)
        print_to_file(f"Number of bkps before checking for significance: {len_bkps}. After checking: {len_bkps_after}")
        pt_list.append(learning_Ids[i])
        
        #find the drastic jumps in the data (glucose at t - glucose at t-1)
        pt_data['time_diff'] = pt_data['dates'].diff()
        pt_data['diff'] = pt_data['glucose_level'].diff()
        glucose_diff = pt_data[(pt_data['time_diff'] == pd.Timedelta(minutes=5))]['diff']
        glucose_diff_list = glucose_diff.tolist()
        
        if (len_bkps_after > 1):
            #get the duration
            duration = pt_data[pt_data['cpd'] == 1]['duration']
            duration_list = duration.tolist()

            #get type and intensity of change
            print_to_file("Get the change type and intensity")
            pt_data, change_type, means, std = type_of_change(pt_data)
            print_to_file(f"Length of the change types, means, std: {len(change_type)} {len(means)} {len(std)}")
            
            #append the type and intensity of change
            all_type_of_change.extend(change_type)
            all_means.extend(means)
            all_std.extend(std)
            all_duration.extend(duration_list)
            all_glucose_diff.extend(glucose_diff_list)

            ## save the files 
            file_name = f"{learning_Ids[i]}.csv"
            file_path = os.path.join(annonated_files,file_name)
            pt_data.to_csv(file_path)

            #append the patient into a big df
            print(pt_data.head(5))
            all_pt = pd.concat([all_pt, pt_data], axis=0).reset_index(drop=True)
            print_to_file(all_pt.tail(5))

            #get features
            print_to_file(f"Number of rows in pt data: {len(pt_data)}")
            X_train, y_train, data = cpd_utils.get_features(pt_data, DATASET)
            print_to_file(f"Number of rows in X_train data: {len(X_train)}. in y_train: {len(y_train)}")
            
            ## save the features to verify correctness 
            file_name = f"{learning_Ids[i]}_features.csv"
            file_path = os.path.join(annonated_files,file_name)
            X_train['PID'] = learning_Ids[i]
            X_train.to_csv(file_path)

            #y_train = data['cpd']
            X_train_total = pd.concat([X_train_total, X_train], axis=0).reset_index(drop=True)
            y_train_total.extend(y_train)
        else:
            print_to_file("Only one breakpoint found")


    #distribution of the type of change
    print_to_file(f"Full length of all: {len(all_pt)}")
    print_to_file(f"Full length. || X_train data: {len(X_train_total)}. in y_train: {len(y_train_total)}")
    type_of_change_list = get_change_type_dist(all_type_of_change)

    #build classifier model for change point detection
    X_train_total['y'] = y_train_total #combine into one df
    X_train_total.to_csv(f"models/{DATASET}_classifier_data.csv")
    print_to_file(all_pt.tail(5))
    #'''#------------------------------------------comment just to run classifier
    X_train_total = pd.read_csv(f"models/{DATASET}_classifier_data.csv")
    print_to_file(f"Total rows for all patients: {len(X_train_total)}")
    #cpd_mdl = build_cpd_model(X_train_total, y_train_total)
    cpd_mdl = build_cpd_model(X_train_total, DATASET)
    print("Training completed")
    
    #save the distribution to pkl
    mdl = {"mdl":cpd_mdl, "type_of_change":type_of_change_list,
    "mean_change": all_means, "std_change": all_std, "duration":all_duration, "glucose_diff":all_glucose_diff}

    # save dictionary to pkl file
    cpd_utils.save_to_pkl(mdl_file, mdl)

    #save the number of changepoints before and after checking for sig
    bkps_df = pd.DataFrame()
    bkps_df['PtID'] = pt_list
    bkps_df['before_checking'] = bkps_before
    bkps_df['after_checking'] = bkps_after
    bkps_df.to_csv(bkps_count_file)
    print_to_file(f"{now}-----------------------------END OF THE RUN- LEARNING.PY")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))