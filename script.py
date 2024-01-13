from sdcclient import IbmAuthHelper, SdMonitorClient

import sys, os
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import datetime
import schedule
import time
import subprocess
import pandas as pd
import numpy as np
import math
from ts import predict
from dl import run_train_lstm


def main():
    services = ["app", "db", "mongo"] # intend to adaptive services
    self_adpt = False # determine whether need begin self-adaptation, True --> adaptation on, False --> adaptation off
    
    ticks = time.time()
    skip_time = 60 # wait 1 mins after each loop
    end = int(ticks)    # current timestamp
    start = end - skip_time   # timestamp of 1 mins ago
    
    init_time = start
    final_time = start + 15*60
    
    total_data = {
        'app': {'data': []},
        'db': {'data': []},
        'mongo': {'data': []}
    }
    all_timestamps = []
    service_uc = {
        'app': [],
        'db': [],
        'mongo': []
    }
    
    cmd = "oc scale deployment app --replicas=5"
    # cmd = "oc scale deployment app --replicas=9"
    output = subprocess.check_output(cmd, shell=True, text=True)

    # for service in services:
    #     cmd = "oc scale deployment {} --replicas=1".format(service)
    #     output = subprocess.check_output(cmd, shell=True, text=True)

    while start < final_time:
        for i, serviceName in enumerate(services):
            if serviceName != "app":
                continue
            print()
            print("Current Timestamp: ", int(ticks))
            cnt = 0
            # Polling each MAPE-K Loop per 1 mins for automatic self-optimizing
            print("Current Service: ", serviceName)
            command = "oc get pods | grep '{}'| grep '{}' | wc -l".format(serviceName,"Running")
            output = subprocess.check_output(command, shell=True, text=True)
        
            pod_count = int(output.strip()) 
        
            # 1 Monitoring
            data_with_keys_and_titles, metric_titles = monitoring(start, start + skip_time, pod_count)
            
            if cnt == 0:
                pre_period = 3600*2
                data_array = np.array([])
                time_array = np.array([])
                pre_data_with_keys_and_titles, metric_titles = monitoring(start-pre_period, start, pod_count)
                data_points = pre_data_with_keys_and_titles['data']+data_with_keys_and_titles['data']
            else: 
                data_points = data_with_keys_and_titles['data']
            
            cnt += 1
            
            data_by_microservice = {} 

            for data_point in data_points:
                microservice = data_point['entry']['kube_deployment_name']

                if microservice not in data_by_microservice:
                    data_by_microservice[microservice] = {
                    'data': []
                    }
                if serviceName == microservice:
                    data_point['entry']['metrics']['pod_count'] = pod_count
                    data_by_microservice[microservice]['data'].append(data_point)
                
                if microservice == 'app':
                    timestamp = data_point['timestamp']
                    metric_value = data_point['entry']['metrics']['sysdig_container_net_connection_in_count']

                    data_array = np.append(data_array, metric_value)
                    time_array = np.append(time_array, timestamp)
                
            # # Create a pandas DataFrame
            # columns = ['time', 'net_count']
            # df = pd.DataFrame(data_list, columns=columns)

            # # Convert timestamp to datetime
            # df['time'] = pd.to_datetime(df['time'], unit='s')

            # # Set timestamp as the index
            # df.set_index('time', inplace=True)
            
            # Combine them into a two-dimensional array
            dataset = np.column_stack((data_array, time_array))
            # predict using LSTM
            future_request_cnt = run_train_lstm(dataset)
            print(future_request_cnt)
            

            if len(data_with_keys_and_titles['data']) == 0:
                print("ERORR: No data to monitor!")
                sys.exit(1)
            else:
                # pod_count = 9
                
                # 2 Analyzing
                uc, est_pods, timestamps = analyzing(data_with_keys_and_titles, pod_count, start, start + skip_time,serviceName, future_request_cnt)
                service_uc[serviceName].append(sum(uc) / len(uc))
                
                # 3 Planning
                cmd, value = planning(uc, est_pods-pod_count)
            
                # 4 Excuting
                if self_adpt:
                    pod_count = excuting(cmd, value, serviceName, pod_count, est_pods)
            
            for servicename, data in data_by_microservice.items():
                if servicename in total_data and len(data['data']):
                    total_data[servicename]['data'].extend(data['data'])
            
        all_timestamps.extend(timestamps)
    
        # Waiting
        # sleep to wait some time to catch next period data
        time.sleep(skip_time - time.time() + ticks)  
        # Compute time difference
        ticks = time.time()
        diff = ticks - end
        if diff < 2 * skip_time:
            end = int(ticks)
            start = end - skip_time
        else:
            print("Timeout! Please re-run this program.")
            break
        
    # plot_metrics(total_data, metric_titles, init_time, final_time)
    
    for i, serviceName in enumerate(services):
        if serviceName == "app":
            total_uc = service_uc[serviceName]
            plot_utility(total_uc, all_timestamps, start, end, serviceName)

def excuting(cmd, num, serviceName, pod_count, est_pods):
    command = "oc get pods | grep '{}'| grep '{}' | wc -l".format(serviceName,"Running")
    output = subprocess.check_output(command, shell=True, text=True)
    count = int(output.strip()) 

    # if(cmd == "U"):
    #     if(count  != pod_count ):
    #         finalPod = count + (pod_count + num - count )
    #     else:
    #         finalPod = count + num 

    # elif (cmd == "D"):
    #     finalPod = pod_count - num  
    # else:
    #     finalPod = count


    print("Current pod is ", str(count), "excepted pod is ", str(est_pods)," current cmd is ", cmd, " opretation pod num ", num)
    
    if cmd == "U" and est_pods != pod_count: 
        est_pods = 10 if est_pods > 10 else est_pods
        scaleUp = "oc scale deployment '{}' --replicas={}".format(serviceName, est_pods)

        try:
            subprocess.run(scaleUp, shell=True, check=True) 
            print("Scale Up, afer scale up the pod number is ", str(est_pods))
        except subprocess.CalledProcessError as e:
            print(f"Scale Up error: {e}")

    elif cmd == "D" and est_pods != pod_count: 
        est_pods = 1 if est_pods < 1 else est_pods
        scaleDown = "oc scale deployment '{}' --replicas={}".format(serviceName, est_pods)

        try:
            subprocess.run(scaleDown, shell=True, check=True) 
            print("Scale Down, afer scale down '{}' the pod number is {}".format(serviceName, est_pods))
        except subprocess.CalledProcessError as e:
            print(f"Scale Down error: {e}")

    elif cmd == "N" or est_pods == pod_count: 
        print("Not need to adaptation")

    # else:
    #     print("Cannot scale beaces pod size == 1 or pod size > 10")
    return est_pods
        

def monitoring(start, end, pod_count, save_json=False):
    # Add the monitoring instance information that is required for authentication
    URL = "https://ca-tor.monitoring.cloud.ibm.com"
    APIKEY = "mEhVtxqB7HTDd5jFCGfHaoMsuUmb2A_H1A6SjvdOHoGV"
    GUID = "b92c514a-ca21-4548-b3f0-4d6391bab407"
    ibm_headers = IbmAuthHelper.get_headers(URL, APIKEY, GUID)
    
    # Instantiate the Python client
    sdclient = SdMonitorClient(sdc_url=URL, custom_headers=ibm_headers)

    # Specify the ID for keys, and ID with aggregation for values
    metrics = [
        {"id": "kube_deployment_name"},
        # 1.Java Runtime
        {"id": "sysdig_container_net_error_count",
        "aggregations": {
            "time": "timeAvg"}},
        {"id": "sysdig_container_net_connection_in_count",
        "aggregations": {
            "time": "timeAvg"}},
    ]

    # Add a data filter or set to None if you want to see "everything"
    filter = "kubernetes.namespace.name=\'acmeair-g12\'"

    # Sampling time:
        #   - for time series: sampling is equal to the "width" of each data point (expressed in seconds)
        #   - for aggregated data (similar to bar charts, pie charts, tables, etc.): sampling is equal to 0
    sampling = 60

    # Paging (from and to included; by default you get from=0 to=9)
    # Here we'll get the top 5.
    paging = {"from": 0, "to": 4}

    # Load data
    ok, res = sdclient.get_data(metrics=metrics,  # List of metrics to query
                                start_ts=start,  # Start of query span is 600 seconds ago
                                end_ts=end,  # End the query span now
                                sampling_s=sampling,  # 1 data point per minute
                                filter=filter,  # The filter specifying the target host
                                # paging=paging,  # Paging to limit to just the 5 most busy
                                datasource_type='container')  # The source for our metrics is the container
    
    if ok:
        # Print summary (what, when)
        start = res['start']
        end = res['end']
        data = res['data']

        print(('Data for %s from %d to %d' % (filter if filter else 'everything', start, end)))
        print('')
        
        # Create a dictionary to hold the data with keys and titles
        data_with_keys_and_titles = {
            'start': start,
            'end': end,
            'data': []
        }

        # Create a list of metric titles
        metric_titles = [metric['id'] for metric in metrics] + ['pod_count']
        metric_new_names = metric_titles[1:][:] + ['pod_count']
        
        # Iterate over the data and add keys and titles
        for d in data:
            timestamp = d['t']
            values = d['d']
            entry = {
                'kube_deployment_name': values[0],
                'metrics': {}
            }
            for i, value in enumerate(values[1:], start=1):
                metric_title = metric_titles[i]
                entry['metrics'][metric_title] = value
            entry['metrics']['pod_count'] = pod_count
            data_with_keys_and_titles['data'].append({
                'timestamp': timestamp,
                'entry': entry
            })
            
        # Convert the response to JSON format
        json_data = json.dumps(data_with_keys_and_titles, indent=4, separators=(',', ': '))

        
        if end:
            # Get the previous timestamp
            timestamp = timestamp_to_datetime(end).strftime('%Y%m%d-%H-%M')
        else:
            # Get current timestamp 
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y%m%d-%H-%M")

        # Save JSON data to file
        if save_json:
            filename = f"{timestamp_to_datetime(start)}_to_{timestamp_to_datetime(end)}.json" 
            with open(filename, "w") as f:
                f.write(json_data)
    else:
        print(res)
        sys.exit(1)

    return data_with_keys_and_titles, metric_titles
    
# Get utility function based on different database
def get_utility_func(database, pod_count):
    max_request = 50
    res = []
    error = []

    # Weight
    # w_pod = 0.3
    # w_err = 0.5
    # w_ncn = 0.2
    
    w_pod = 0.3
    w_ncn = 0.4
    
    # # U_c=w_(Pod num)·p_(Pod num)+w_(net_connection_cnt)·p_(net_connection_cnt)+w_(hhtp_error )·p_(hhtp_error)
    # for item in database:
    #     net_connection = item['metrics']['sysdig_container_net_connection_in_count']
    #     net_error = item['metrics']['sysdig_container_net_error_count']
    #     cur_uc = cal_pod_count(pod_count) * w_pod + cal_sysdig_container_net_connection_in_count(net_connection) * w_ncn + cal_sysdig_container_net_error_count(net_error) * w_err
    #     print("net_error_count ", net_error)
    #     res.append(float(cur_uc))
    #     error.append(float((net_error)))
    # return res, error
    est_pod = pod_count
    
    for item in database:
        net_connection = item
        workloadRatio = net_connection/(max_request*pod_count)
        cur_uc = cal_workloadRatio(workloadRatio) * w_ncn

        res.append(float(cur_uc))
    res = np.array(res)
    avg_request = sum(database) / len(database)
    # est_pod = math.ceil(avg_request / max_request)
    est_pod = round(avg_request / max_request)
    est_pod = 1 if est_pod < 1 else est_pod
    res += cal_pod_count(pod_count) * w_pod + cal_pod_diff(abs(est_pod-pod_count)) * 0.3
    return res, est_pod
    


def cal_pod_count(x):
    if x >= 9:
        return 0.5
    elif x >= 6 and x < 9:
        return 0.6
    elif x >= 3 and x < 6:
        return 0.7
    elif x > 1 and x < 3:
        return 0.8
    else:
        return 1.0

def cal_pod_diff(x):
    if x >= 5:
        return 0
    elif x >= 3 and x < 5:
        return 0.1
    elif x >= 2 and x < 3:
        return 0.2
    elif x >= 1 and x < 2:
        return 0.5
    else:
        return 1.0

def cal_workloadRatio(x):
    if x < 0.2:
        return 1.0
    elif x >= 0.2 and x < 0.5:
        return 0.9
    elif x >= 0.5 and x < 0.7:
        return 0.8
    elif x >= 0.7 and x < 0.95:
        return 0.7
    elif x >= 0.95 and x < 1.1:
        return 0.5
    else:
        return 0
    
def cal_sysdig_container_net_connection_in_count(x):
    if x > 400:
        return 1.0
    elif x >= 150 and x < 400:
        return 0.8
    elif x >= 100 and x < 150:
        return 0.5
    elif x >= 50 and x < 100:
        return 0.3
    elif x >= 10 and x < 50:
        return 0.2
    else:
        return 0

def cal_sysdig_container_net_error_count(x):
    if x > 100:
        return 0
    elif x > 50 and x <= 100:
        return 0.1
    elif x > 30 and x <= 50:
        return 0.3
    elif x >  10 and x <= 30:
        return 0.9
    elif x <= 10:
        return 1.0

def planning(uc, count_diff):
    # Calculate average utility
    average_uc = sum(uc) / len(uc)
    
    # return cmd and value for controling during excuting stage
    cmd = 'N'   # N: Nothing    U: Scale-up     D: Scale-down
    value = 0   # decreasing/increasing the number of pods, from 0 to 2 (if cmd == 'N' then value == 0)
    
    print(f"Average utility: {average_uc}")
    
    # Recommend actions based on utility level
    cmd = 'U' if count_diff > 0 else 'D'
    if count_diff == 0:
        cmd = 'N'
    value = abs(count_diff)
    # if average_uc < 0.35:
    #     print("Bad!")
    #     cmd = 'U' if count_diff > 0 else 'D'
    
    # elif average_uc >= 0.35 and average_uc < 0.65:
    #     cmd = 'U' if count_diff > 0 else 'D'
    #     if count_diff == 0:
    #         cmd = 'N'
    # else:  
    #     print("Well done! ")
    return cmd, value

# Function to convert timestamp to datetime 
def timestamp_to_datetime(ts):
  return datetime.datetime.fromtimestamp(ts)

def plot_metrics(total_data, metric_titles, start, end):
    if True:
        metric_new_names = metric_titles[1:]
        # Get current timestamp 
        now = datetime.datetime.now()
        # Create metrics folder if not exists
        figures_folder = 'plots'
        os.makedirs(figures_folder, exist_ok=True)
        for i, metric in enumerate(metric_titles[1:]):
            fig, ax = plt.subplots()
            yLable = ""
            for microservice, data in total_data.items():
                if len(data['data']) == 0:
                    continue
                # Convert timestamps 
                timestamps = [d['timestamp'] for d in data['data']]
                # Convert timestamps 
                datetimes = []
                values = []
                for ts, d in zip(timestamps, data['data']):
                    dt = timestamp_to_datetime(ts)
                    if dt.minute % 1 == 0:
                        datetimes.append(dt.strftime('%H:%M'))
                        values.append(d['entry']['metrics'][metric])
                ax.plot(datetimes, values, label=microservice)

            ax.legend()
            ax.set_xticklabels(datetimes, rotation=45)

            fig.set_size_inches(19,6)
            ax.set_title(metric_new_names[i])
            if(metric == "sysdig_container_net_error_count"):
                ax.set_ylabel("Count")
            elif(metric == "sysdig_container_net_connection_in_count"):
                ax.set_ylabel("Count")
            else:
                ax.set_ylabel("Count")
            ax.set_xlabel("Time")
            plt.savefig(os.path.join(figures_folder, f'{metric}_{timestamp_to_datetime(start)}_to_{timestamp_to_datetime(end)}.png'))
            
            # Close plot 
            plt.close()

def plot_utility(uc_values, timestamps, start, end, serviceName):
    figures_folder = 'utilities'
    os.makedirs(figures_folder, exist_ok=True)
    fig, ax = plt.subplots()

    # Plot utility value against timestamp 
    ax.plot(timestamps, uc_values)

    ax.set_ylabel("Utility Value")
    ax.set_xlabel("Timestamp")

    # Format x-axis labels 
    datetimes = [timestamp_to_datetime(t).strftime('%H:%M') for t in timestamps] 
    # ax.set_xticks(range(len(datetimes)))
    ax.set_xticklabels(datetimes, rotation=45)
    
    from matplotlib.ticker import FixedLocator

    locator = FixedLocator(timestamps)
    ax.xaxis.set_major_locator(locator)

    # Add title, save figure
    ax.set_title("Utility of " + serviceName)
    fig.set_size_inches(19,6)
    # fig.set_size_inches(25, 10)
    plt.savefig(os.path.join(figures_folder,f'utility_function_{timestamp_to_datetime(start)}_{timestamp_to_datetime(end)}_{serviceName}.png'))

    # Close plot
    plt.close()

def analyzing(data_with_keys_and_titles, pod_count, start, end, serviceName, future_request_cnt):
    # Get the most significant dataset
    service_metrics = []

    for item in data_with_keys_and_titles['data']:
        if item['entry']['kube_deployment_name'] == serviceName:
            metrics = item['entry']['metrics']
            timestamp = item['timestamp']
    
            service_metrics.append({
                'timestamp': timestamp, 
                'metrics': metrics
            })
    
    # Calculate the Uc function
    # future_request_cnt = [10, 20, 30, 20, 10]
    uc, est_pods = get_utility_func(future_request_cnt, pod_count)
    # has_error = True if sum(error)/len(error) > 30 else False

    # Plot
    timestamps = [item['timestamp'] for item in service_metrics]
    
    return list(uc), est_pods, timestamps

# def analyzing(future_net_count, pod_count, start, end, serviceName):
#     # Get the most significant dataset
#     service_metrics = []

#     # for item in data_with_keys_and_titles['data']:
#     #     if item['entry']['kube_deployment_name'] == serviceName:
#     #         metrics = item['entry']['metrics']
#     #         timestamp = item['timestamp']
    
#     #         service_metrics.append({
#     #             'timestamp': timestamp, 
#     #             'metrics': metrics
#     #         })
    
#     # Calculate the Uc function
#     uc, error = get_utility_func(service_metrics, pod_count)
#     has_error = True if sum(error)/len(error) > 30 else False

#     # Plot
#     timestamps = [item['timestamp'] for item in service_metrics]
    
#     return uc, timestamps, has_error

if __name__ == "__main__":
    main()
