import pandas as pd
import numpy as np
import time, glob, os
import pickle
from numpy import mean
import sys, os, math, time
import logging
#from datetime import datetime
from time import gmtime, strftime
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import string
from matplotlib.lines import Line2D
import tsfel
from datetime import datetime, timedelta
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm, uniform, expon
from pylab import *
from scipy.optimize import curve_fit
import eda_utils
import warnings

from logging.config import dictConfig
import json
with open('logging.json', 'r') as read_file:
    contents = json.load(read_file)
dictConfig(contents)
LOGGER = logging.getLogger()

warnings.filterwarnings('ignore')


# Calculate the log-likelihood of the data given the bimodal distribution
def neg_log_likelihood_bimodal(params, data):
    mu1, sigma1, A1, mu2, sigma2, A2 = params
    likelihood = bimodal(data, mu1, sigma1, A1, mu2, sigma2, A2)
    neg_log_likelihood = -np.sum(np.log(likelihood))
    return neg_log_likelihood

# Define the bimodal function
def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return A1 * np.exp(-(x - mu1) ** 2 / (2 * sigma1 ** 2)) + A2 * np.exp(-(x - mu2) ** 2 / (2 * sigma2 ** 2))

# Calculate the log-likelihood of the data given a distribution
def log_likelihood(data, dist_name, params,distribution):
    if dist_name == 'bimodal':
        return -np.sum(np.log(bimodal(data, *params)))
    else:
        return np.sum(distribution.logpdf(data, *params))


def sample_from_dist(data_to_fit, sample_size, dist_name, params): #(data_to_fit, sample_size, property_name=''):
    #dist_name, params = get_dist(data_to_fit,data_name=property_name )
    #seed_value = 42
    #np.random.seed(seed_value)
    param1 = params[0]
    param2 = params[1]
    if (dist_name == 'expon'): #scale and rate
        #data = np.random.exponential(scale=1/param1, size=sample_size)
        data = np.random.exponential(scale=param2, size=sample_size)
        data = np.ceil(data).astype(int)
    elif(dist_name == 'uniform'): #lower and upper bound
        data = np.random.uniform(param1, param2, sample_size)
    elif(dist_name == 'norm'): #mean and std
        data = np.random.normal(param1, param2, sample_size)
    elif(dist_name == 'bimodal'):
        data = np.random.choice(data_to_fit, size=sample_size, p=bimodal(data_to_fit,*params) / sum(bimodal(data_to_fit, *params)))
    #LOGGER.debug(f"Distribution: {dist_name} || Params: {params} || Sampled data: {data}")
    return data

def predict_cpd_duration(file, mdl_dict,filename):
    LOGGER.debug("Predicting changepoint")
    ## fit the duration to a distrbution
    dist_name, params = eda_utils.get_dist(mdl_dict['duration'],data_name='')
    df = pd.read_csv(file)
    df['PtID'] = filename
    df['cpd'] = 0
    mdl = mdl_dict['mdl']

    index_arr = []
    duration_arr = []

    window_size = 10 #10, 5seconds #15
    freq = 4
    window_size = window_size * freq
    # cfg_file = tsfel.load_json("features_simple.json")
    cfg_file = tsfel.get_features_by_domain()
    i = window_size
    while i < len(df):
        X_train = pd.Series([])
        cpd_cnt = 0
        t2 = i - window_size
        #LOGGER.debug(f"index number in this pt sample: {t2+1}:{i}")
        window = df[t2:i]['eda_signal'].values

        X_train = tsfel.time_series_features_extractor(
                    cfg_file,
                    window,
                    fs=freq,                 # freq = 4 earlier in the function
                    verbose=0,
                    n_jobs=0)

        need_cols = mdl.feature_names_in_    # available in XGBoost ≥1.7
        X_train = X_train.reindex(columns=need_cols, fill_value=0)

        #LOGGER.debug(len(window))
        # X_train = tsfel.time_series_features_extractor(cfg_file, window, n_jobs = 15, verbose = 0)
        #LOGGER.debug(f"X_train: {X_train}")
        y_pred = mdl.predict(X_train)
        if(y_pred == 1):
            #LOGGER.debug(f"CPD Found")
            cpd_cnt = cpd_cnt + 1
            index_arr.append(i) #array of cpd index
            dur = sample_from_dist(mdl_dict['duration'], 1, dist_name, params)[0] #sample duration
            duration_arr.append(dur) #seconds
            i = i + int(dur*freq)
        else:
            i = i+1
    cpd_details = pd.DataFrame()
    cpd_details['CPD_Index'] = index_arr
    cpd_details['duration'] = duration_arr
    return cpd_details

def predict_change_properties(file, mdl_dict, filename, cpd_details_folder):
    cpd_details = predict_cpd_duration(file, mdl_dict, filename)
    types = mdl_dict['type_of_change']['type']
    types_prop = mdl_dict['type_of_change']['prob']
    LOGGER.debug(f'types: {types}')
    LOGGER.debug(f'types_prop: {types_prop}')

    ### Sample the type and size of change
    bkps_len = len(cpd_details)
    LOGGER.debug(f"Total cpd in this file : {bkps_len} {(cpd_details)}")
    type_change = np.random.choice(a=types,size=bkps_len,p=types_prop)

    dist_name, params = eda_utils.get_dist(mdl_dict['mean_change'],data_name='mean_change')
    mean_change_drawn = sample_from_dist(mdl_dict['mean_change'],bkps_len, dist_name, params)

    dist_name, params = eda_utils.get_dist(mdl_dict['std_change'],data_name='')
    std_change_drawn = sample_from_dist(mdl_dict['std_change'], bkps_len, dist_name, params)

    cpd_details['type_of_change'] = type_change
    cpd_details['mean_change'] = mean_change_drawn
    cpd_details['std_change'] = std_change_drawn

    #save the cpd_details - TO DO
    file_name = f"{filename}.csv"
    file_path = os.path.join(cpd_details_folder,file_name)
    os.makedirs(cpd_details_folder, exist_ok=True)
    cpd_details.to_csv(file_path, index=False)

    #get the df
    df = pd.read_csv(file)
    df['PtID'] = filename
    #merge it with the df
    #df_final = pd.merge(df, cpd_details, right_on='CPD_Index', left_index=True, how='left')
    df_final = df.merge(cpd_details.set_index('CPD_Index'), left_index=True, right_index=True, suffixes=('', '_cpd'), how='left')
    df_final.loc[df_final['duration'].notna(), 'cpd'] = 1

    LOGGER.debug(f"CPD count: {len(df_final[df_final['cpd']==1])}")
    LOGGER.debug(f"len of the df: {len(df_final)}")

    return df_final

def insert_values(df, index_list, values_list,col_name):
    for i in range(len(index_list)):
        df.at[index_list[i], col_name] = values_list[i]
    return df

def insert_one_value(df, index, value,col_name):
    df.at[index, col_name] = value
    return df


def modify_values(df, threshold):
    min_value = df['eda_signal'].min()
    max_value = df['eda_signal'].max()
    LOGGER.debug(f"Min value: {min_value}. Max value: {max_value}")
    df['eda_signal_new'] = df['eda_signal']
    df['mean_change_new'] = df['mean_change']
    df['std_change_new'] = df['std_change']
    df_new = pd.DataFrame()

    bkps = df[df['cpd'] == 1].index #list of the breakpoints index
    LOGGER.debug(df[df['cpd'] == 1])
    LOGGER.debug(f"{len(bkps)} changepoints: {bkps}")
    for i in range(len(bkps)):
        LOGGER.debug(f'-----------------------Modifying values: index: {i}: bkp_index: {bkps[i]}')
        #LOGGER.debug(len(df))
        #LOGGER.debug(df.index)
        #LOGGER.debug(df.iloc[bkps[i]])
        change_type =  df.loc[bkps[i], 'type_of_change'] #df['type_of_change'].iloc[bkps[i]]
        change_mu = df['mean_change'].iloc[bkps[i]]
        change_sigma = df['std_change'].iloc[bkps[i]]
        change_duration = df['duration'].iloc[bkps[i]]
        LOGGER.debug(f'Sample change in mean: {change_mu}. Sample change in std: {change_sigma}. Duration: {change_duration}')

        start_gradual = bkps[i] #df['dates'].iloc[bkps[i]] #index of the start
        end_gradual = start_gradual + int(change_duration * 4) #start_gradual + timedelta(minutes=int(change_duration))
        LOGGER.debug(f"Start_gradual:{start_gradual} || End Gradual:{end_gradual}")
        LOGGER.debug(f"Start_gradual:{start_gradual} || End Gradual:{end_gradual}")

        if(len(bkps) == 1):
            pre_bkp = df[0:bkps[i]]
            post_bkp = df[bkps[i]:len(df)]
        elif(i == 0):
            pre_bkp = df[0:bkps[i]]
            post_bkp = df[bkps[i]:bkps[i+1]]
        elif(i == len(bkps)-1):
            pre_bkp = df[bkps[i-1]:bkps[i]]
            post_bkp = df[bkps[i]:len(df)]
        else:
            pre_bkp = df[bkps[i-1]:bkps[i]]
            post_bkp = df[bkps[i]:bkps[i+1]]
        previous_value = df.at[bkps[i]-1, 'eda_signal_new']
        change_properties = [change_mu, change_sigma,change_duration]
        threshold_values = [min_value, max_value, threshold]
        post_bkp_new_orig, post_bkp_new, std_change_new = modify_property(change_type,post_bkp,change_properties,pre_bkp,threshold_values,previous_value)
        if (len(post_bkp_new) != 0):
            df = insert_values(df, post_bkp.index, post_bkp_new_orig,'eda_signal_new_original')
            df = insert_values(df, post_bkp.index, post_bkp_new,'eda_signal_new')
            df = insert_one_value(df, post_bkp.index[0],std_change_new,'eda_signal_new')
    df_new = df_new.append(df)
    LOGGER.debug(f"total df length: {len(df_new)}")
    return df_new


def gradual_mean_change(post_ckp, duration_array, mu1, mu2, duration):
    new_vals = []
    for i in range(len(duration_array)):
        time_passed = pd.Timedelta(duration_array[i] - duration_array[0]).total_seconds()/60.0
        time_passed = 1 if time_passed == 0 else time_passed
        if (time_passed > duration):
            new_val = post_ckp[i] + (mu2 - mu1)
        else:
            new_val = post_ckp[i] + ((mu2 - mu1)/duration) * time_passed
        new_vals.append(new_val)
    return new_vals

def get_new_std(sigma1, sigma2):
    ###check if sigma2 (desired std) is not too far from sigma1(current std). i.e it shouldnt be > 50% increase of sigma1
    if (sigma2 > (1.5*sigma1)):
        sigma2 = 1.5*sigma1
    return sigma2

def modify_property(change_type, post_bkp, change_properties, pre_bkp, threshold_values, previous_value):
    change_mu = change_properties[0]
    change_sigma = change_properties[1]
    duration = change_properties[2]
    post_ckp = post_bkp['eda_signal_new'].tolist() #the glucose values after the changepoints
    duration_array = post_bkp.index.tolist() #the dates after the changepoints
    pre_mean = np.mean(pre_bkp['eda_signal_new'])
    LOGGER.debug(f"pre_mean: {pre_mean}")
    pre_sigma = np.std(pre_bkp['eda_signal_new'])
    min_value = threshold_values[0] #min glucose value
    max_value = threshold_values[1] #max glucose value
    threshold = threshold_values [2] #the threshold for the maximum  drastic diff btw cosecutive measurements
    std_change_new = np.nan

    mu1 = np.mean(post_ckp) #post_bkp_mean  #initial mean
    sigma1 = 0.001 if (np.std(post_ckp) == 0) else np.std(post_ckp)
    sigma2 = 0.001

    mu2 = mu1 + change_mu if (np.isnan(pre_mean)) else (pre_mean + change_mu) #desired mean

    post_ckp_new = []
    LOGGER.debug(f"change_mu:{change_mu} | duration:{duration} | dur_len: {len(duration_array)}")
    LOGGER.debug(f"post_mean_old_new: {mu1}_{mu2} | mean_adjust: {(mu2-mu1)} | pre_mean: {pre_mean} | change_type: {change_type}")

    post_ckp_arr = np.array(post_ckp)
    if (change_type=='mean+std'):
        sigma2 = pre_sigma + change_sigma
        sigma2_new = get_new_std(sigma1, sigma2)
        change_sigma_new = sigma2_new - pre_sigma
        post_ckp_arr = ((post_ckp_arr - mu1) * (sigma2_new/sigma1)) + mu1
        mu1 =  np.mean(post_ckp_arr) #recalculate the initial mean before shifting mean
        post_ckp_new_original = gradual_mean_change(post_ckp_arr.tolist(), duration_array,mu1,mu2,duration)
    elif(change_type=='mean'):
        post_ckp_new_original = gradual_mean_change(post_ckp, duration_array, mu1, mu2, duration)
        sigma2_new = sigma2
    elif(change_type=='std'):
        sigma2 = pre_sigma + change_sigma
        sigma2_new = get_new_std(sigma1, sigma2)
        change_sigma_new = sigma2_new - pre_sigma
        post_ckp_new_original = (((post_ckp_arr-mu1)*(sigma2_new/sigma1)) + mu1).tolist()
    LOGGER.debug(f"post_sigma1: {sigma1} | post_sigma2_old: {sigma2} | post_sigma2: {sigma2_new} | presigma: {pre_sigma} | division_old: {(sigma2/sigma1)} | division: {(sigma2_new/sigma1)}")

    for i in range(len(duration_array)): #ensure it's not < or > min/max eda
        new_value_original = post_ckp_new_original[i]
        new_value = new_value_original
        if new_value < min_value:
            LOGGER.debug("Value too low")
            new_value = min_value
        elif new_value > max_value:
            LOGGER.debug("Value too high")
            new_value = max_value
        #LOGGER.debug(f"index: {i} | time_passed: {time_passed} | change: {change} | old_value: {post_ckp[i]} | new_value: {new_value}")
        LOGGER.debug(f"initial_value: {post_ckp[i]} | new_value_original: {new_value_original} | new_value: {new_value} | prev_value: {previous_value}")
        post_ckp_new.append(new_value)
    return post_ckp_new_original, post_ckp_new, change_sigma_new
def main():
    global DATASET
    DATASET = str(sys.argv[1]) #get the type of data
    input_folder = str(sys.argv[2]) + "*_eda.csv"
    output_folder = str(sys.argv[3])
    mdl_file =str(sys.argv[4])
    cpd_details_folder = str(sys.argv[5])
    threshold = str(sys.argv[6])

    LOGGER.debug(f'{input_folder=}, {output_folder=}, {mdl_file=}, {cpd_details_folder=}, {threshold=}')

    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    LOGGER.debug(f"{now}-----------------------------NOW STARTING A NEW RUN- SIMULATING.PY")


    mdl_dict = pickle.load(open(mdl_file, 'rb'))
    for file in glob.glob(input_folder):
        filename2 = os.path.basename(file)
        filename = os.path.splitext(filename2)[0].split('_')[0]
        LOGGER.debug(f"{file}----{filename}")
        LOGGER.debug(f"Processing: {file}")
        sim_data = predict_change_properties(file, mdl_dict,filename,cpd_details_folder)
        LOGGER.debug("Done predicting change properties. Start modifying the values")
        sim_data = modify_values(sim_data, threshold)

        #rename cols
        new_column_names = {'eda_signal': 'eda_signal_old', 'eda_signal_new':'eda_signal'}
        # Rename the columns
        sim_data.rename(columns=new_column_names, inplace=True)
        #save to the output folder
        #output_path = output_folder + filename2
        output_path = os.path.join(output_folder,filename2)
        os.makedirs(output_folder, exist_ok=True)
        sim_data.to_csv(output_path, index=False)
    LOGGER.debug("Done processing all files! Yipee!!!!")


if __name__ == "__main__":
    LOGGER.debug("Started running")
    start_time = time.time()
    main()
    LOGGER.debug("--- %s seconds ---" % (time.time() - start_time))
    LOGGER.debug("Ended --- %s seconds ---" % (time.time() - start_time))