import pandas as pd
import numpy as np
import time, glob, os
import cpd_utils_old2
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

def print_to_file(statement):
    file = f'debug_{DATASET}.txt'

    logging.basicConfig(filename=file, level=logging.DEBUG, format='')
    
    logging.debug(statement)

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

def get_dist(data, save_file, data_name='', xlabel='Value', plot_graph=False):
    # Define a list of candidate distributions to fit
    candidate_distributions = [stats.uniform, stats.norm, stats.expon]
    if (data_name == 'mean_change'):#only one that looks like bimodal
        candidate_distributions.append('bimodal_pdf')
    best_fit = None
    best_fit_params = {}
    best_fit_loglikelihood = -np.inf
    plt.figure(figsize=(8, 5))
    # Iterate through candidate distributions and fit to data using MLE
    for distribution in candidate_distributions:
        # Fit the distribution to the data using MLE
        if distribution == 'bimodal_pdf':
            y, x = np.histogram(data, bins=50)
            # Calculate the bin centers from the bin edges
            x = (x[1:] + x[:-1]) / 2
           
            # Provide initial guesses for the parameters of the bimodal distribution
            initial_params = (np.mean(data), np.std(data), 50, np.mean(data), np.std(data), 50)
            params, _ = curve_fit(bimodal,x, y, initial_params)
            distribution_name = 'bimodal'
            bimodal_params = params
        else:
            params = distribution.fit(data, loc=np.mean(data), scale=np.std(data))
            distribution_name = distribution.name 
        loglikelihood = log_likelihood(data, distribution_name, params, distribution)

        print(f"Distribution: {distribution_name} || Log-Likelihood: {loglikelihood} || Params: {params}")
        
        # Check if this distribution provides a better fit
        if loglikelihood > best_fit_loglikelihood:
            best_fit = distribution
            best_fit_params = params
            best_fit_loglikelihood = loglikelihood
            best_fit_name = distribution_name

    if plot_graph:
        # Plot the histogram of the data
        #plt.hist(data, bins=100, density=True, alpha=0.5, label='Data')
        y, x = np.histogram(data, bins=100, density=True)
        x_mid = (x[1:] + x[:-1]) / 2
        plt.bar(x_mid, y, width=(x[1] - x[0]), color='skyblue', alpha=0.5)

        # Define a colormap for the distributions
        colormap = {
            stats.norm: 'orange',
            stats.uniform: 'green',
            stats.expon: 'red',
            'bimodal_pdf': 'blue'
        }

        # Plot the best-fitting distribution
        x = np.linspace(np.min(data), np.max(data), 100)
        if best_fit == 'bimodal_pdf':
            bimodal_curve = bimodal(x, *bimodal_params)
            plt.plot(x, bimodal_curve * y.max() / bimodal_curve.max(), 'r-', color=colormap[best_fit], label=f'Best fit: {distribution_name}')
        else:
            plt.plot(x, best_fit.pdf(x, *best_fit_params), 'r-', color=colormap[best_fit], label=f'Best fit: {best_fit.name}')

        # Plot the other candidate distributions
        for distribution in candidate_distributions:
            print(distribution)
            if distribution == best_fit:
                continue
            if (distribution == 'bimodal_pdf' and best_fit != 'bimodal_pdf'):
                bimodal_curve = bimodal(x, *bimodal_params)
                plt.plot(x, bimodal_curve * y.max() / bimodal_curve.max(), 'r-', color=colormap[distribution], label=f'{distribution_name}')
            elif (distribution != best_fit):
                params = distribution.fit(data, loc=np.mean(data), scale=np.std(data))
                plt.plot(x, distribution.pdf(x, *params), color=colormap[distribution], label=f"{distribution.name}")
                
        # Set plot labels and legend
        plt.xlabel(xlabel, fontsize=22)
        plt.ylabel('Probability Density', fontsize=22)
        plt.xticks(fontsize=22)             # Adjust the font size of the X ticks
        plt.yticks(fontsize=22) 
        plt.legend(fontsize=18)
        #plt.savefig(save_file, dpi=400)
        plt.show()

    # Print the best-fitting distribution and its parameters
    print(f"Best fit: {best_fit_name}")
    print(f"Parameters: {best_fit_params}")
    #print_to_file(f"Distribution: {best_fit_name} || Params: {best_fit_params}")
    return best_fit_name, best_fit_params

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
    #print_to_file(f"Distribution: {dist_name} || Params: {params} || Sampled data: {data}")
    return data

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

def predict_cpd_duration(file, mdl_dict,filename):
    ## fit the duration to a distrbution
    dist_name, params = get_dist(mdl_dict['duration'],data_name='')
    df = pd.read_csv(file)
    df['PtID'] = filename
    df = df.rename(columns={'dateString': 'dates'})
    df['dates'] = pd.to_datetime(df['dates'])
    df['cpd'] = 0
    mdl = mdl_dict['mdl']
    min_duration = 5 #minutes
   

    date_arr = []
    duration_arr = []

    window_size = 60 #120 #minutes
    daily_list = group_into_daily(df)
    cfg_file = tsfel.load_json("features_simple.json")
    for i in range(len(daily_list)):#len(daily_list)
        print(f"Processing Day {i} data in file: {filename} ")
        per_day = daily_list[i]
        t1 = pd.Timestamp(per_day['dates'].iloc[12]) #per_day['dates'].iloc[i] #start from the 12th values
        #print(f"T1 {t1}----Last line-{per_day['dates'].iloc[-1]}")
        cpd_cnt = 0
        while (t1 < per_day['dates'].iloc[-1]):
            t2 = t1 - pd.to_timedelta(window_size, unit='m')
            #print_to_file(f"T1: {t1} || T2: {t2}")
            window = per_day[(per_day['dates'] >= t2) & (per_day['dates'] < t1)]['glucose_level'].values
            #print(f"Window: {window}. Extract features")
            X_train = tsfel.time_series_features_extractor(cfg_file, window, n_jobs = 3, verbose = 0)
            #np.random.seed(42) 
            y_pred = mdl.predict(X_train)
            if(y_pred == 1):
                cpd_cnt = cpd_cnt + 1
                date_arr.append(t1)
                dur = int(sample_from_dist(mdl_dict['duration'], 1, dist_name, params)[0])#sample duration
                if (dur < min_duration):
                    print_to_file (f"Dur sampled {dur} is too small: set to {min_duration}")
                    dur = min_duration
                #elif (dur > max_duration):
                    #print_to_file (f"Dur sampled {dur} is too large: set to {max_duration}")
                    #dur = max_duration
                #if dur is larger than time left in the day
                if ((t1+timedelta(minutes=dur)) >  per_day['dates'].iloc[-1]):
                    dur =  per_day['dates'].iloc[-1] - t1
                    print(f"Overshoot: {dur} {per_day['dates'].iloc[-1]} {t1}")
                    dur = dur.total_seconds() / 60
                    print(f"Overshoot2: {dur}")
                duration_arr.append(dur)
                time_delay = timedelta(minutes=dur)
                t1 = t1+time_delay
                #print(f"New T1 before checking if it is in the df: {t1}")
                # Check if the updated t1 exists in the DataFrame
                if t1 in per_day['dates'].values:
                    #print("it's in table")
                    t1 = t1  # No change needed, as t1+time_delay exists in the DataFrame
                else:
                    # Find the next timestamp in per_day['dates'] that is greater than t1
                    next_timestamps = per_day['dates'][per_day['dates'] > t1]
                    if not next_timestamps.empty:
                        t1 = next_timestamps.iloc[0]  # Update t1 with the next timestamp
                #print(f"New T1 after checking if it is in the df: {t1}. Dur sampled: {dur}")
                #print_to_file(f"Duration sampled: {dur}----Start time for next prediction: {t1}")
            else:
                #t1 = t1+timedelta(minutes=sample_rate)
                next_timestamps = per_day['dates'][per_day['dates'] > t1]
                if not next_timestamps.empty:
                    t1 = next_timestamps.iloc[0]  # Update t1 with the next timestamp
            #print(f"No of cpd found: {cpd_cnt}")
    cpd_details = pd.DataFrame()
    cpd_details['dates'] = date_arr
    cpd_details['duration'] = duration_arr
    return cpd_details

def predict_change_properties(file, mdl_dict, filename, cpd_details_folder):
    cpd_details = predict_cpd_duration(file, mdl_dict, filename)
    types = mdl_dict['type_of_change']['type']
    types_prop = mdl_dict['type_of_change']['prob']
    print_to_file(types)
    print_to_file(types_prop)
    
    ### Sample the type and size of change
    bkps_len = len(cpd_details)
    print_to_file(f"Total cpd in this file: {bkps_len}")
    type_change = np.random.choice(a=types,size=bkps_len,p=types_prop)
    
    dist_name, params = get_dist(mdl_dict['mean_change'],data_name='mean_change')
    mean_change_drawn = sample_from_dist(mdl_dict['mean_change'],bkps_len, dist_name, params)
    
    dist_name, params = get_dist(mdl_dict['std_change'],data_name='')
    std_change_drawn = sample_from_dist(mdl_dict['std_change'], bkps_len, dist_name, params) 
    
    cpd_details['type_of_change'] = type_change
    cpd_details['mean_change'] = mean_change_drawn
    cpd_details['std_change'] = std_change_drawn

    #save the cpd_details - TO DO 
    file_name = f"{filename}"
    file_path = os.path.join(cpd_details_folder,file_name)
    os.makedirs(cpd_details_folder, exist_ok=True) #create the folder if it doesnt exist
    cpd_details.to_csv(file_path)

    #get the df
    df = pd.read_csv(file)
    df['PtID'] = filename
    df = df.rename(columns={'dateString': 'dates'})
    df['dates'] = pd.to_datetime(df['dates'])

    #merge it with the df
    cpd_details['dates'] = pd.to_datetime(cpd_details['dates'])
    df_final = pd.merge(df, cpd_details, on='dates', how='left')
    df_final.loc[df_final['duration'].notna(), 'cpd'] = 1

    print_to_file(f"CPD count: {len(df_final[df_final['cpd']==1])}")

    return df_final

def sample_with_bounds(property_name, len_of_samples, mdl_dict,lower_bound,upper_bound):
    dist_name, params = get_dist(mdl_dict[property_name],data_name=property_name)
    change_drawn = sample_from_dist_with_bounds(mdl_dict[property_name],len_of_samples, dist_name, params,lower_bound, upper_bound)
    return change_drawn

def sample_from_dist_with_bounds(data_to_fit, sample_size, dist_name, params, lower_bound, upper_bound): 
    param1 = params[0]
    param2 = params[1]
    for i in range (3):
        sample = sample_from_dist(data_to_fit, sample_size, dist_name, params)
        if lower_bound <= sample <= upper_bound:
            return sample
    return np.clip(sample, lower_bound, upper_bound)
    
def insert_values(df, index_list, values_list,col_name):
    for i in range(len(index_list)):
        df.at[index_list[i], col_name] = values_list[i] 
    return df

def insert_one_value(df, index, value,col_name):
    df.at[index, col_name] = value
    return df

def modify_values(sim_data, threshold):
    min_value = sim_data['glucose_level'].min()
    max_value = sim_data['glucose_level'].max()
    print(f"Min value: {min_value}. Max value: {max_value}")
    sim_data['glucose_level_new'] = sim_data['glucose_level']
    sim_data['mean_change_new'] = sim_data['mean_change']
    sim_data['std_change_new'] = sim_data['std_change']
    df_new = pd.DataFrame()

    #bkps = df[df['cpd'] == 1].index #list of the breakpoints index 
    daily_list = group_into_daily(sim_data)
    for j in range(len(daily_list)): #len(daily_list)
        print(f"\nModfying values in Day {j} data")
        per_day = daily_list[j]
        df = per_day.reset_index()
        #display(df)
        bkps = df[df['cpd'] == 1].index #list of the breakpoints index 
        print(f"{len(bkps)} changepoints in Day {j}: {bkps}")
        for i in range(len(bkps)):
            print(f'-----------------------Modifying values: index: {i}: bkp_index: {bkps[i]} in Day {j}')
            change_type = df['type_of_change'].iloc[bkps[i]]
            change_mu = df['mean_change'].iloc[bkps[i]]
            change_sigma = df['std_change'].iloc[bkps[i]]
            change_duration = df['duration'].iloc[bkps[i]]
            #print_to_file(f'Sample change in mean: {change_mu}. Sample change in std: {change_sigma}. Duration: {change_duration}')
            print_to_file(f'Sample change in mean: {change_mu}. Sample change in std: {change_sigma}. Duration: {change_duration}')

            start_gradual = df['dates'].iloc[bkps[i]]
            end_gradual = start_gradual + timedelta(minutes=int(change_duration))
            mask = (df['dates'] >= start_gradual) & (df['dates'] <= end_gradual)
            end_gradual_index = df.loc[mask].index[-1]
            print_to_file(f"Start_gradual:{start_gradual} || End Gradual:{end_gradual} || last_index:{end_gradual_index}")

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
            previous_value = df.at[bkps[i]-1, 'glucose_level_new']
            change_properties = [change_mu, change_sigma,change_duration]
            threshold_values = [min_value, max_value, threshold]
            post_bkp_new_orig, post_bkp_new, std_change_new = modify_property(change_type,post_bkp,change_properties,pre_bkp,threshold_values,previous_value)
            if (len(post_bkp_new) != 0):
                df = insert_values(df, post_bkp.index, post_bkp_new_orig,'glucose_level_new_original')
                df = insert_values(df, post_bkp.index, post_bkp_new,'glucose_level_new')
                df = insert_one_value(df, post_bkp.index[0],std_change_new,'std_change_new')
        df_new = df_new.append(df)
    print(f"total df length: {len(df_new)}")
    return df_new

def get_new_std(sigma1, sigma2):
    ###check if sigma2 (desired std) is not too far from sigma1(current std)
    if (sigma2 > (1.5*sigma1)):
        sigma2 = 1.5*sigma1
    return sigma2

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

def modify_property(change_type, post_bkp, change_properties, pre_bkp, threshold_values, previous_value):
    change_mu = change_properties[0]
    change_sigma = change_properties[1]
    duration = change_properties[2]
    post_ckp = post_bkp['glucose_level_new'].tolist() #the glucose values after the changepoints
    duration_array = post_bkp['dates'].tolist() #the dates after the changepoints
    pre_mean = np.mean(pre_bkp['glucose_level_new'])
    print_to_file(f"pre_mean: {pre_mean}")
    pre_sigma = np.std(pre_bkp['glucose_level_new'])
    min_value = threshold_values[0] #min glucose value
    max_value = threshold_values[1] #max glucose value 
    threshold = threshold_values [2] #the threshold for the maximum  drastic diff btw cosecutive measurements
    std_change_new = np.nan

    mu1 = np.mean(post_ckp) #post_bkp_mean  #initial mean
    sigma1 = 0.001 if (np.std(post_ckp) == 0) else np.std(post_ckp)
    sigma2 = 0.001

    mu2 = mu1 + change_mu if (np.isnan(pre_mean)) else (pre_mean + change_mu) #desired mean

    post_ckp_new = []
    print_to_file(f"change_mu:{change_mu} | duration:{duration} | dur_len: {len(duration_array)}")
    print_to_file(f"post_mean_old_new: {mu1}_{mu2} | mean_adjust: {(mu2-mu1)} | pre_mean: {pre_mean} | change_type: {change_type}")

    post_ckp_arr = np.array(post_ckp)
    if (change_type=='mean+std'):
        sigma2 = pre_sigma + change_sigma
        sigma2_new = get_new_std(sigma1, sigma2)
        change_sigma_new = sigma2_new - pre_sigma
        #post_ckp_arr = post_ckp_arr * (sigma2_new/sigma1)
        post_ckp_arr = ((post_ckp_arr - mu1) * (sigma2_new/sigma1)) + mu1
        mu1 =  np.mean(post_ckp_arr)
        post_ckp_new_original = gradual_mean_change(post_ckp_arr.tolist(), duration_array,mu1,mu2,duration)
    elif(change_type=='mean'):
        post_ckp_new_original = gradual_mean_change(post_ckp, duration_array, mu1, mu2, duration)
    elif(change_type=='std'):
        sigma2 = pre_sigma + change_sigma
        sigma2_new = get_new_std(sigma1, sigma2)
        change_sigma_new = sigma2_new - pre_sigma
        #post_ckp_new_original = (post_ckp_arr*(sigma2_new/sigma1)).tolist()
        post_ckp_new_original = (((post_ckp_arr-mu1)*(sigma2_new/sigma1)) + mu1).tolist()
    print_to_file(f"post_sigma1: {sigma1} | post_sigma2_old: {sigma2} | post_sigma2: {sigma2_new} | presigma: {pre_sigma} | division_old: {(sigma2/sigma1)} | division: {(sigma2_new/sigma1)}")

    for i in range(len(duration_array)):
        #apply the drastic jump check
        if (i > 0):
            previous_value = post_ckp_new[i-1]   
        new_value_original = post_ckp_new_original[i]
        diff = drastic_jump_diff(threshold, previous_value,new_value_original)
        new_value = new_value_original + diff
        
        #ensure it's not < or > min/max glucose
        if new_value < min_value:
            print_to_file("Value too low")
            new_value = min_value
        elif new_value > max_value:
            print_to_file("Value too high")
            new_value = max_value
        
        #print_to_file(f"time_passed: {time_passed} | change: {change} | ")
        print_to_file(f"initial_value: {post_ckp[i]} | new_value_original: {new_value_original} | new_value: {new_value} | prev_value: {previous_value}")
        post_ckp_new.append(new_value)
    return post_ckp_new_original, post_ckp_new, change_sigma_new

def get_first_new_value(change_type, post_ckp, change_mu, change_sigma,
                    duration, duration_array, pre_mean, pre_sigma, min_value, max_value):
    mu1 = np.mean(post_ckp) #post_bkp_mean  #initial mean
    sigma1 = 0.001 if (np.std(post_ckp) == 0) else np.std(post_ckp)
    sigma2 = 0.001
        
    if np.isnan(pre_mean):
        adjustment = change_mu
    else:
        mu2 = pre_mean + change_mu #desired mean
        adjustment = mu2 - mu1
    for i in range(1):
        time_passed = pd.Timedelta(duration_array[i] - duration_array[0]).total_seconds()/60.0
        time_passed = 1 if time_passed == 0 else time_passed
        if (time_passed > duration):
            change = adjustment
        else:
            change = (adjustment/duration) * time_passed
        
        if (change_type=='mean+std'):
            sigma2 = pre_sigma + change_sigma
            new_value = post_ckp[i] + (change * (sigma2/sigma1))
        elif(change_type=='mean'):
            new_value = post_ckp[i] + change
        elif(change_type=='std'):
            sigma2 = pre_sigma + change_sigma
            new_value = post_ckp[i] + (sigma2/sigma1)
            
        if new_value < min_value:
            new_value = min_value
        elif new_value > max_value:
            new_value = max_value
        print("First Value")
        print(f"index: {i} | time_passed: {time_passed} | change: {change} | sigma2: {sigma2} | sigma1: {sigma1} | old_value: {post_ckp[i]} | new_value: {new_value}")
    return new_value

def drastic_jump_diff(threshold, previous_value,new_value_first):
    threshold_high = threshold #from the results-analysis notebook
    threshold_low = threshold * -1
    drastic_jump = previous_value - new_value_first
    diff = 0
    if (drastic_jump > threshold_high):
        #subtract the diff to have the maximum drastic jump 
        print_to_file(f"Drastic jump found!!! {drastic_jump}. y(t-1) was: {previous_value}. y(t): {new_value_first}")
        diff = previous_value - threshold_high - new_value_first
    elif(drastic_jump < threshold_low):
        print_to_file(f"Drastic jump found!!! {drastic_jump}. y(t-1) was: {previous_value}. y(t): {new_value_first}")
        diff = previous_value - threshold_low - new_value_first
    return diff


def main():
    global DATASET
    DATASET = str(sys.argv[1]) #get the type of data
    input_folder = str(sys.argv[2]) + "*.csv"
    output_folder = str(sys.argv[3])
    mdl_file =str(sys.argv[4])
    cpd_details_folder = str(sys.argv[5])
    threshold = np.float64(sys.argv[6])
    
    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print_to_file(f"{now}-----------------------------NOW STARTING A NEW RUN- SIMULATING.PY")
    
    print(DATASET)
        
    mdl_dict = pickle.load(open(mdl_file, 'rb'))
    for file in glob.glob(input_folder):
        filename = os.path.basename(file)
        print_to_file(f"Processing: {file}")
        sim_data = predict_change_properties(file, mdl_dict,filename,cpd_details_folder)
        print_to_file("Done predicting change properties. Start modifying the values")
        sim_data = modify_values(sim_data, threshold)
        
        #rename cols 
        new_column_names = {'dates': 'dateString','glucose_level': 'glucose_level_old', 'glucose_level_new':'glucose_level'}
        # Rename the columns
        sim_data.rename(columns=new_column_names, inplace=True)
        #save to the output folder 
        output_path = output_folder + filename
        os.makedirs(output_folder, exist_ok=True)
        sim_data.to_csv(output_path)
    print_to_file("Done processing all files!")


if __name__ == "__main__":
    print("Started running")
    start_time = time.time()
    main()
    print_to_file("--- %s seconds ---" % (time.time() - start_time))
    print("Ended --- %s seconds ---" % (time.time() - start_time))