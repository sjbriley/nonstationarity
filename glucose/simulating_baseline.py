import pandas as pd
import numpy as np
import random
import time, glob, sys, os, math, time
import datetime, logging
from datetime import timedelta


def print_to_file(statement):
    file = f'debug_{DATASET}_baseline.txt'

    logging.basicConfig(filename=file, level=logging.DEBUG, format='')
    
    logging.debug(statement)


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

def generate_durations(indices,dates, dur_array, df, dist_type):
    print(f"Index to generate dur: {indices}")
    durations = []
    for i in range(len(indices)):
        print(f'Index: {indices[i]}')
        if(i == len(indices)-1):
            distance = (df.iloc[-1]['dates'] - dates[i]).total_seconds()/60
        else:
            distance = (dates[i + 1] - dates[i]).total_seconds()/60
        '''
        if (dist_type == '1'):
            #duration = random.randint(dur_array[0], dur_array[1])
            duration = dur_array[-1]
        elif (dist_type == '2'):
            duration = abs(int(np.random.normal(dur_array[2], dur_array[3],1)[0]))
            #duration = np.random.exponential(scale=dur_array[4], size=1)
        '''
        duration = dur_array[-1]
        if (duration < 0):
            print_to_file("duration < 0")
        if (duration > distance):
            print_to_file(f"Sampled duration {duration} > distance {distance}. ")
            duration = distance #random.randint(min_duration, distance) 
        durations.append(duration)
    return durations


def generate_indices(data_len, mindist, cpd_count):
    '''
    Genrate the random indices for the chnagepoints
    such that min duration is maintained between the generated changepoints
    '''
    range_size = data_len - ((mindist - 1) * (cpd_count -1)) 
    print(range_size)
    #return [(mindist-1)*i + x for i, x in enumerate(sorted(random.sample(range(range_size), cpd_count)))]
    return [(mindist-1)*i + x for i, x in enumerate(sorted(random.sample(range(1, range_size), cpd_count)))]


def predict_cpd_duration(file,filename, cpd_count_arr, duration, dist_type):
    ## fit the duration to a distrbution
    df = pd.read_csv(file)
    df['PtID'] = filename
    df = df.rename(columns={'dateString': 'dates'})
    df['dates'] = pd.to_datetime(df['dates'])
    df['cpd'] = 0
    
    min_duration = duration[-1] #set to the static duration to avoid overlap of the changepoints
    date_arr = []
    duration_arr = []

    daily_list = group_into_daily(df)
    for i in range(len(daily_list)):
        dates = []
        print(f"Processing Day {i} data in file: {filename}")
        per_day = daily_list[i].reset_index()
        print(len(per_day))

        if ((dist_type == '1')  or (dist_type == '2')):
            cpd_count = cpd_count_arr[-1]
        elif(dist_type == '3' or (dist_type == '4')):
            cpd_count = int(np.random.normal(cpd_count_arr[2], cpd_count_arr[3], 1)[0])
        print(f'cpd count: {cpd_count}')
        if (cpd_count < 0):
            print_to_file("cpd < 0")
        index_arr = generate_indices(len(per_day),int(min_duration/5)+2, cpd_count+1) #[random.randint(0, len(per_day)) for _ in range(cpd_count)]
        index_arr = index_arr[:cpd_count] 

        for i in index_arr:
            dates.append(pd.Timestamp(per_day['dates'].iloc[i]))
        dur = generate_durations(index_arr, dates, duration, per_day, dist_type) 
        duration_arr.extend(dur)
        date_arr.extend(dates)
    cpd_details = pd.DataFrame()
    cpd_details['dates'] = date_arr
    cpd_details['duration'] = duration_arr
    print_to_file(cpd_details)
    return cpd_details

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
    df = df.rename(columns={'dateString': 'dates'})
    df['dates'] = pd.to_datetime(df['dates'])

    #merge it with the df
    cpd_details['dates'] = pd.to_datetime(cpd_details['dates'])
    df_final = pd.merge(df, cpd_details, on='dates', how='left')
    df_final.loc[df_final['duration'].notna(), 'cpd'] = 1
    print_to_file(f"CPD count: {len(df_final[df_final['cpd']==1])}")
    return df_final

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
    print_to_file(f"Min value: {min_value}. Max value: {max_value}")
    sim_data['glucose_level_new'] = sim_data['glucose_level']
    sim_data['glucose_level_new_original'] = sim_data['glucose_level']
    sim_data['std_change_new'] = sim_data['std_change']
    df_new = pd.DataFrame()

    daily_list = group_into_daily(sim_data)
    for j in range(len(daily_list)): #len(daily_list)
        print_to_file(f"\nModfying values in Day {j} data")
        per_day = daily_list[j]
        df = per_day.reset_index()
        bkps = df[df['cpd'] == 1].index #list of the breakpoints index 
        print_to_file(f"{len(bkps)} changepoints in Day {j}: {bkps}")
        for i in range(len(bkps)):
            print(f'-----------------------Modifying values: index: {i}: bkp_index: {bkps[i]} in Day {j}')
            change_type = df['type_of_change'].iloc[bkps[i]]
            change_mu = df['mean_change'].iloc[bkps[i]]
            change_sigma = df['std_change'].iloc[bkps[i]]
            change_duration = df['duration'].iloc[bkps[i]]
            print_to_file(f'Change in mean: {change_mu}. Change in std: {change_sigma}. Duration: {change_duration}')

            start_gradual = df['dates'].iloc[bkps[i]]
            end_gradual = start_gradual + timedelta(minutes=int(change_duration))
            mask = (df['dates'] >= start_gradual) & (df['dates'] <= end_gradual)
            end_gradual_index = df.loc[mask].index[-1]
            #gradual = df.loc[mask].reset_index()
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
            #print(bkps[i]-1)
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
    post_ckp = post_bkp['glucose_level_new'].tolist() #the glucose values after the changepoints
    duration_array = post_bkp['dates'].tolist() #the dates after the changepoints
    pre_mean = np.mean(pre_bkp['glucose_level_new'])
    print_to_file(f"pre_mean: {pre_mean}")
    pre_sigma = np.std(pre_bkp['glucose_level_new'])
    min_value = threshold_values[0] #min glucose value
    max_value = threshold_values[1] #max glucose value 
    threshold = threshold_values [2] #the threshold for the maximum  drastic diff btw cosecutive measurements
    change_sigma_new = np.nan

    mu1 = np.mean(post_ckp) #post_bkp_mean  #initial mean
    sigma1 = 0.001 if (np.std(post_ckp) == 0) else np.std(post_ckp)
    sigma2 = 0.001
    sigma2_new = 0.001

    mu2 = mu1 + change_mu if (np.isnan(pre_mean)) else (pre_mean + change_mu) #desired mean

    post_ckp_new = []
    print_to_file(f"change_mu:{change_mu} | duration:{duration} | dur_len: {len(duration_array)}")
    print_to_file(f"post_mean_old_new: {mu1}_{mu2} | mean_adjust: {(mu2-mu1)} | pre_mean: {pre_mean} | change_type: {change_type}")

    post_ckp_arr = np.array(post_ckp)
    if (change_type=='mean+std'):
        sigma2 = pre_sigma + change_sigma
        sigma2_new = get_new_std(sigma1, sigma2)
        change_sigma_new = sigma2_new - pre_sigma
        post_ckp_arr = ((post_ckp_arr - mu1) * (sigma2_new/sigma1)) + mu1 #post_ckp_arr * (sigma2_new/sigma1)
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
            #new_value = post_ckp[i] + (change * (sigma2/sigma1))
            new_value = (post_ckp[i] + change)* (sigma2/sigma1)
        elif(change_type=='mean'):
            new_value = post_ckp[i] + change
        elif(change_type=='std'):
            sigma2 = pre_sigma + change_sigma
            new_value = post_ckp[i] * (sigma2/sigma1)
            
        if new_value < min_value:
            new_value = min_value
        elif new_value > max_value:
            new_value = max_value
        print_to_file(f"First Value | time_passed: {time_passed} | change: {change} | sigma2: {sigma2} | sigma1: {sigma1} | old_value: {post_ckp[i]} | new_value: {new_value}")
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
    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    #print_to_file(f"{now}-----------------------------NOW STARTING A NEW RUN- SIMULATING.PY")
    
    if (DATASET=='oaps'):
        #TO DO include ohio data link here
        print(DATASET)
        input_folder = str(sys.argv[2])
        output_folder = str(sys.argv[3]) 
        cpd_details_folder = str(sys.argv[4])
        cpd_count = [0,7,3,1,2] #last item in the array is the random value used for baseline1
        duration = [5, 1335,20,5,209,30] #[min, max, mean, std, random_value]
        mean_change =  [-226.88, 183.88, 0, 10,5] 
        std_change =  [-93.50, 95.56, 0, 1,0.5] 
        threshold = 20
        dist_type = '2'

    elif(DATASET=='ohio'):
        #TO DO include ohio data link here
        print(DATASET)
        input_folder = str(sys.argv[2])
        output_folder = str(sys.argv[3]) 
        cpd_details_folder = str(sys.argv[4])
        cpd_count = [0,7,3,1,2] #last item in the array is the random value used for baseline1 and 2
        duration = [5, 1320,20,5,209,30] #[min, max, mean, std, exp_param2]
        mean_change =  [-226.88, 183.88, 0, 10,5] 
        std_change =  [-93.50, 95.56, 0, 1,0.5] 
        threshold = 40
        dist_type = '2' #baseline1 {fixed value..mean shift only}, baseline2{fixed values, mean+std}, baseline3, baseline4
        
    print_to_file(f"generating results for {output_folder}")
    for file in glob.glob(input_folder):
        filename = os.path.basename(file)
        print_to_file(f"Processing: {file}")
        sim_data = predict_change_properties(file,filename,cpd_details_folder,cpd_count, duration, mean_change, std_change, dist_type)
        print_to_file("Done predicting change properties. Start modifying the values")
        sim_data = modify_values(sim_data, threshold)
        
        #rename cols 
        new_column_names = {'dates': 'dateString','glucose_level': 'glucose_level_old', 'glucose_level_new':'glucose_level'}
        # Rename the columns
        sim_data.rename(columns=new_column_names, inplace=True)
        output_path = output_folder + filename
        os.makedirs(output_folder, exist_ok=True)
        sim_data.to_csv(output_path)
    print_to_file("Done processing all files! Yipee!!!!")


if __name__ == "__main__":
    print("Started running")
    start_time = time.time()
    main()
    print_to_file("--- %s seconds ---" % (time.time() - start_time))
    print("Ended --- %s seconds ---" % (time.time() - start_time))