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
import random

def print_to_file(statement):
    file = f'debug_{DATASET}_baseline.txt'

    logging.basicConfig(filename=file, level=logging.DEBUG, format='')
    
    logging.debug(statement)

def generate_durations(indices, freq, dur_array, df_length):
    print_to_file(indices)
    durations = []
    for i in range(len(indices)):
        if(i == len(indices)-1):
            distance = (df_length - indices[i])/freq #distance between the cpd_x and cpd_x+1
        else:
            distance = (indices[i + 1] - indices[i])/freq
        #duration = round(random.uniform(min_duration, max_duration), 2)
        duration = dur_array[-1]
        print_to_file(f"duration = {duration} - distance = {distance}")
        if (duration > distance):
            print_to_file(f"Sampled duration {duration} > distance {distance}.")
            duration = distance
        durations.append(duration)
    return durations

def predict_cpd_duration_1(file, filename, cpd_count, min_duration, max_duration):
    print("Predicting changepoint")
    df = pd.read_csv(file)
    df['PtID'] = filename
    df['cpd'] = 0
    freq = 4
    #index_arr = [random.randint(0, len(df)) for _ in range(cpd_count)]
    index_arr = random.sample(range(len(df)), cpd_count) #random unique values
    index_arr.sort()
    duration_arr = generate_durations(index_arr, freq,  min_duration, max_duration, len(df)) 
    cpd_details = pd.DataFrame()
    cpd_details['CPD_Index'] = index_arr
    cpd_details['duration'] = duration_arr
    return cpd_details

def generate_indices(data_len, mindist, cpd_count):
    '''
    Genrate the random indices for the chnagepoints
    such that min duration is maintained between the generated changepoints
    '''
    range_size = data_len - ((mindist - 1) * (cpd_count -1)) 
    print(range_size)
    #return [(mindist-1)*i + x for i, x in enumerate(sorted(random.sample(range(range_size), cpd_count)))]
    return [(mindist-1)*i + x for i, x in enumerate(sorted(random.sample(range(1, range_size), cpd_count)))]

def predict_cpd_duration(file, filename, cpd_count_arr, duration, dist_type):
    print("Predicting changepoint")
    df = pd.read_csv(file)
    df['PtID'] = filename
    df['cpd'] = 0
    freq = 4
    min_duration = duration[-1]

    if ((dist_type == '1')  or (dist_type == '2')):
        cpd_count = cpd_count_arr[-1]
    elif(dist_type == '3' or (dist_type == '4')):
        cpd_count_ = int(np.random.normal(cpd_count_arr[2], cpd_count_arr[3], 1)[0])
        cpd_count = max(0, cpd_count_) #ensure only psotive values
    print(f'cpd count: {cpd_count}')
    if (cpd_count < 0):
        print_to_file("cpd < 0")

    index_arr = generate_indices(len(df),int(min_duration*freq)+2, cpd_count+1) #[random.randint(0, len(per_day)) for _ in range(cpd_count)]
    index_arr = index_arr[:cpd_count] 
    
    index_arr.sort()
    duration_arr = generate_durations(index_arr, freq, duration, len(df)) 
    cpd_details = pd.DataFrame()
    cpd_details['CPD_Index'] = index_arr
    cpd_details['duration'] = duration_arr
    return cpd_details

def predict_change_properties_1(file, filename, cpd_details_folder, cpd_count, min_duration, max_duration):
    cpd_details = predict_cpd_duration(file, filename, cpd_count, min_duration, max_duration)
    types = ['mean+std'] #mdl_dict['type_of_change']['type']
    types_prop = [1] #mdl_dict['type_of_change']['prob']
    print_to_file(types)
    print_to_file(types_prop)
    
    ### Sample the type and size of change
    bkps_len = len(cpd_details)
    print_to_file(f"Total cpd in this file : {bkps_len} {(cpd_details)}")
    type_change = np.random.choice(a=types,size=bkps_len)
    
    mean_change_drawn =  np.round(np.random.uniform(-0.8, 1.7, size=bkps_len), 1) #used 10 to 50 b4
    std_change_drawn = np.round(np.random.uniform(-0.7, 0.8, size=bkps_len), 1)
    
    cpd_details['type_of_change'] = type_change
    cpd_details['mean_change'] = mean_change_drawn
    cpd_details['std_change'] = std_change_drawn

    #save the cpd_details - TO DO 
    file_name = f"{filename}.csv"
    file_path = os.path.join(cpd_details_folder,file_name)
    os.makedirs(cpd_details_folder, exist_ok=True)
    cpd_details.to_csv(file_path)

    #get the df
    df = pd.read_csv(file)
    df['PtID'] = filename
    #merge it with the df
    #df_final = pd.merge(df, cpd_details, right_on='CPD_Index', left_index=True, how='left')
    df_final = df.merge(cpd_details.set_index('CPD_Index'), left_index=True, right_index=True, suffixes=('', '_cpd'), how='left')
    df_final.loc[df_final['duration'].notna(), 'cpd'] = 1

    print_to_file(f"CPD count: {len(df_final[df_final['cpd']==1])}")
    print(f"len of the df: {len(df_final)}")

    return df_final


def predict_change_properties(file, filename, cpd_details_folder,cpd_count, duration, mean_change, std_change, dist_type):
    cpd_details = predict_cpd_duration(file, filename, cpd_count, duration, dist_type)
    bkps_len = len(cpd_details)
    print_to_file(f"Total cpd in this file: {bkps_len}")
    std_change_drawn = np.nan
    mean_change_drawn = np.nan
    if (dist_type == '1'):
        mean_change_drawn = [mean_change[-1]] * bkps_len
        types = ['mean']
    elif (dist_type == '2'):
        mean_change_drawn = [mean_change[-1]] * bkps_len
        print(mean_change_drawn)
        std_change_drawn = [std_change[-1]] * bkps_len
        print(std_change_drawn)
        types = ['mean+std']
    elif(dist_type == '3'):
        mean_change_drawn = np.round(np.random.normal(mean_change[2], mean_change[3], size=bkps_len), 1)
        types = ['mean']
    elif(dist_type == '4'):
        mean_change_drawn = np.round(np.random.normal(mean_change[2], mean_change[3], size=bkps_len), 1)
        std_change_drawn = np.round(np.random.normal(std_change[2], std_change[3], size=bkps_len), 1)
        types = ['mean+std']

    type_change = types * bkps_len 
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
    #df = df.rename(columns={'dateString': 'dates'})
    #df['dates'] = pd.to_datetime(df['dates'])

    #merge it with the df
    df_final = df.merge(cpd_details.set_index('CPD_Index'), left_index=True, right_index=True, suffixes=('', '_cpd'), how='left')
    df_final.loc[df_final['duration'].notna(), 'cpd'] = 1
    print_to_file(f"CPD count: {len(df_final[df_final['cpd']==1])}")
    print(f"len of the df: {len(df_final)}")
    return df_final

def insert_values(df, index_list, values_list,col_name):
    for i in range(len(index_list)):
        df.at[index_list[i], col_name] = values_list[i] 
    return df
def insert_one_value(df, index, value,col_name):
    df.at[index, col_name] = value
    return df

def modify_property_1(change_type, post_ckp, change_mu, change_sigma,
                    duration, duration_array, pre_mean, pre_sigma, min_value, max_value, previous_value):
    print(f"changtype: {change_type}")
    mu1 = np.mean(post_ckp) #post_bkp_mean  #initial mean
    sigma1 = 0.001 if (np.std(post_ckp) == 0) else np.std(post_ckp)
    sigma2 = 0.001
    
    if np.isnan(pre_mean):
        adjustment = change_mu
    else:
        mu2 = pre_mean + change_mu #desired mean
        adjustment = mu2 - mu1
    #print(f"mu1: {mu1} | mu2: {mu2} | pre_mean: {pre_mean}")
    post_ckp_new = []
    post_ckp_after_gradual_new = []
    mean_change_new_list = []
    #print("---------In modify property function-------")
    #print(f"change_mu:{change_mu} | duration:{duration} | adjustment: {adjustment} | len_of_dur: {len(duration_array)}")
        
    for i in range(len(duration_array)):
        #print(f'handling new value {i}')

        after_gradual = False
        time_passed = duration_array[i] - duration_array[0] #no of rows that have passed
        time_passed = 1 if time_passed == 0 else time_passed
        if (time_passed > int(duration * 4)): #4 is the freq
            change = adjustment
            after_gradual = True
            #print('After gradual')
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

        #print(f"index: {i} | time_passed: {time_passed} | change: {change} | old_value: {post_ckp[i]} | new_value: {new_value}")
        post_ckp_new.append(new_value)
        mean_change_new_list.append(change_mu)
        if (after_gradual == True):
            post_ckp_after_gradual_new.append(new_value)
    print(f"The length is {len(post_ckp_after_gradual_new)}")
    return post_ckp_new, post_ckp_after_gradual_new, mean_change_new_list

def modify_values_1(df, freq):
    min_value = df['eda_signal'].min()
    max_value = df['eda_signal'].max()
    print(f"Min value: {min_value}. Max value: {max_value}")
    df['eda_signal_new'] = df['eda_signal']
    df['mean_change_new'] = df['mean_change']
    df_new = pd.DataFrame()

    bkps = df[df['cpd'] == 1].index #list of the breakpoints index 
    print(df[df['cpd'] == 1])
    print(f"{len(bkps)} changepoints: {bkps}")
    for i in range(len(bkps)):
        print(f'-----------------------Modifying values: index: {i}: bkp_index: {bkps[i]}')
        #print(len(df))
        #print_to_file(df.index)
        #print(df.iloc[bkps[i]])
        change_type =  df.loc[bkps[i], 'type_of_change'] #df['type_of_change'].iloc[bkps[i]]
        change_mu = df['mean_change'].iloc[bkps[i]]
        change_sigma = df['std_change'].iloc[bkps[i]]
        change_duration = df['duration'].iloc[bkps[i]]
        print(f'Sample change in mean: {change_mu}. Sample change in std: {change_sigma}. Duration: {change_duration}')

        start_gradual = bkps[i] #df['dates'].iloc[bkps[i]] #index of the start 
        end_gradual = start_gradual + int(change_duration * freq) #start_gradual + timedelta(minutes=int(change_duration))
        print_to_file(f"Start_gradual:{start_gradual} || End Gradual:{end_gradual}")
        print(f"Start_gradual:{start_gradual} || End Gradual:{end_gradual}")

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
        pre_bkp_mean = mean(pre_bkp['eda_signal_new'])
        pre_sigma = np.std(pre_bkp['eda_signal_new'])
        post_bkp_mean = mean(post_bkp['eda_signal_new'])
        post_sigma = np.std(post_bkp['eda_signal_new'])
        post_bkp_new,post_ckp_after_gradual_new,mean_change_list = modify_property(change_type,
                                                                        post_bkp['eda_signal_new'].tolist(),
                                                                    change_mu, change_sigma,
                                        change_duration, post_bkp.index.tolist(), pre_bkp_mean, pre_sigma,
                                        min_value, max_value, previous_value)
        
        print(f'Pre chkp mean:{pre_bkp_mean} || Old postchkp mean: {post_bkp_mean} || New postchkp mean:{mean(post_ckp_after_gradual_new)} || Previous value: {previous_value}.')

        if (len(post_bkp_new) != 0):
            df = insert_values(df, post_bkp.index, post_bkp_new,'eda_signal_new')
            df = insert_values(df, post_bkp.index, mean_change_list,'mean_change_new')
    df_new = df_new.append(df)
    print(f"total df length: {len(df_new)}")
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
    post_ckp = post_bkp['eda_signal_new'].tolist() #the eda values after the changepoints
    duration_array = post_bkp.index.tolist() #post_bkp['dates'].tolist() #the dates after the changepoints
    pre_mean = np.mean(pre_bkp['eda_signal_new'])
    print_to_file(f"pre_mean: {pre_mean}")
    pre_sigma = np.std(pre_bkp['eda_signal_new'])
    min_value = threshold_values[0] #min eda value
    max_value = threshold_values[1] #max eda value 
    #threshold = threshold_values [2] #the threshold for the maximum  drastic diff btw cosecutive measurements
    change_sigma_new = np.nan

    mu1 = np.mean(post_ckp) #post_bkp_mean  #initial mean
    sigma1 = 0.001 if (np.std(post_ckp) == 0) else np.std(post_ckp)
    sigma2 = 0.001
    sigma2_new = 0.001
    
    # #if np.isnan(pre_mean):
    #    # adjustment = change_mu
    # else:
    #     mu2 = pre_mean + change_mu #desired mean
    #     adjustment = mu2 - mu1
    #print(f"mu1: {mu1} | mu2: {mu2} | pre_mean: {pre_mean}")
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

    for i in range(len(duration_array)): #ensure it's not < or > min/max eda
        new_value_original = post_ckp_new_original[i]
        new_value = new_value_original
        if new_value < min_value:
            print_to_file("Value too low")
            new_value = min_value
        elif new_value > max_value:
            print_to_file("Value too high")
            new_value = max_value    

        #print(f"index: {i} | time_passed: {time_passed} | change: {change} | old_value: {post_ckp[i]} | new_value: {new_value}")
        print_to_file(f"initial_value: {post_ckp[i]} | new_value_original: {new_value_original} | new_value: {new_value} | prev_value: {previous_value}")
        post_ckp_new.append(new_value)
    return post_ckp_new_original, post_ckp_new, change_sigma_new

def modify_values(df, freq):
    min_value = df['eda_signal'].min()
    max_value = df['eda_signal'].max()
    print(f"Min value: {min_value}. Max value: {max_value}")
    df['eda_signal_new'] = df['eda_signal']
    df['std_change_new'] = df['std_change']
    df_new = pd.DataFrame()

    bkps = df[df['cpd'] == 1].index #list of the breakpoints index 
    print(df[df['cpd'] == 1])
    print(f"{len(bkps)} changepoints: {bkps}")
    for i in range(len(bkps)):
        print(f'-----------------------Modifying values: index: {i}: bkp_index: {bkps[i]}')
        #print(len(df))
        #print_to_file(df.index)
        #print(df.iloc[bkps[i]])
        change_type =  df.loc[bkps[i], 'type_of_change'] #df['type_of_change'].iloc[bkps[i]]
        change_mu = df['mean_change'].iloc[bkps[i]]
        change_sigma = df['std_change'].iloc[bkps[i]]
        change_duration = df['duration'].iloc[bkps[i]]
        print(f'Sample change in mean: {change_mu}. Sample change in std: {change_sigma}. Duration: {change_duration}')

        start_gradual = bkps[i] #df['dates'].iloc[bkps[i]] #index of the start 
        end_gradual = start_gradual + int(change_duration * freq) #start_gradual + timedelta(minutes=int(change_duration))
        print_to_file(f"Start_gradual:{start_gradual} || End Gradual:{end_gradual}")
        print(f"Start_gradual:{start_gradual} || End Gradual:{end_gradual}")

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
        threshold_values = [min_value, max_value]
        post_bkp_new_orig, post_bkp_new, std_change_new = modify_property(change_type,post_bkp,change_properties,pre_bkp,threshold_values,previous_value)
            
        #print(f'Pre chkp mean:{pre_bkp_mean} || Old postchkp mean: {post_bkp_mean} || New postchkp mean:{mean(post_ckp_after_gradual_new)} || Previous value: {previous_value}.')
        if (len(post_bkp_new) != 0):
            df = insert_values(df, post_bkp.index, post_bkp_new_orig,'eda_signal_new_original')
            df = insert_values(df, post_bkp.index, post_bkp_new,'eda_signal_new')
            df = insert_one_value(df, post_bkp.index[0],std_change_new,'std_change_new')
            #df = insert_values(df, post_bkp.index, mean_change_list,'mean_change_new')
    df_new = df_new.append(df)
    print(f"total df length: {len(df_new)}")
    return df_new

def main():
    global DATASET
    DATASET = str(sys.argv[1]) #get the type of data - WESAD


    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print_to_file(f"{now}-----------------------------NOW STARTING A NEW RUN- SIMULATING.PY")
    
    if (DATASET=='WESAD'):
        #TO DO include ohio data link here
        print(DATASET)
        input_folder = str(sys.argv[2]) 
        output_folder = str(sys.argv[3]) 
        cpd_details_folder = str(sys.argv[4])
        cpd_count = [0,7,20,10,20] #last item in the array is the random value used for baseline1
        duration = [0, 0,0,0,0,10] #[min, max, mean, std, random_value]#seconds
        mean_change =  [-226.88, 183.88, 0, 1,1] 
        std_change =  [-93.50, 95.56, 0.1, 0.01,0.5] 
        dist_type = '4'
        freq=4
        
    
    for file in glob.glob(input_folder):
        filename2 = os.path.basename(file)
        filename = os.path.splitext(filename2)[0].split('_')[0]
        print(f"{file}----{filename}")
        print_to_file(f"Processing: {file}")
        sim_data = predict_change_properties(file,filename,cpd_details_folder, cpd_count, duration, mean_change, std_change, dist_type)
        print_to_file("Done predicting change properties. Start modifying the values")
        sim_data = modify_values(sim_data,freq)
    
        #rename cols 
        new_column_names = {'eda_signal': 'eda_signal_old', 'eda_signal_new':'eda_signal'}
        # Rename the columns
        sim_data.rename(columns=new_column_names, inplace=True)
        #save to the output folder 
        #output_path = output_folder + filename2 
        output_path = os.path.join(output_folder,filename2)
        os.makedirs(output_folder, exist_ok=True) 
        print(output_path)
        sim_data.to_csv(output_path)
    print_to_file("Done processing all files!")


if __name__ == "__main__":
    print("Started running")
    start_time = time.time()
    main()
    print_to_file("--- %s seconds ---" % (time.time() - start_time))
    print("Ended --- %s seconds ---" % (time.time() - start_time))