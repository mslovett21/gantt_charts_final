import pandas as pd
import numpy as np
import gc
import glob
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.0f}'.format


############################ SET VARIABLES ####################################
SPLIT_FILES_DIR = "../split_files/"
IMG_DESTINATION_DIR = "/gantt_chart_images/"
MY_DPI = 90

parent_path          = os.path.dirname(os.getcwd())
img_destination_path = parent_path + IMG_DESTINATION_DIR

# COLUMNS SETS needed for the program
cols_drop = ['Unnamed: 0', 'pre_script_start', 'pre_script_end', 'type' ]

columns_names = ["ready_delay","wms_delay",'pre_script_delay', "queue_delay", 
                 'stage_in_delay',"runtime", 'stage_out_delay',"post_script_delay","finished"]


def preprocess_df(file_path, cols_drop):   
    gantt_chart = pd.read_csv(file_path)
    gantt_chart = gantt_chart.drop(columns = cols_drop)
    gantt_chart = gantt_chart.fillna(0)
    return gantt_chart

def add_delays(gantt_chart_df):
    
    gantt_chart_df["post_script_delay"] = gantt_chart_df["post_script_end"] - gantt_chart_df["post_script_start"]
    first_wf_task                    = gantt_chart_df["ready"][0]
    gantt_chart_df["ready_delay"]    = gantt_chart_df["ready"] - first_wf_task
    gantt_chart_delays               = gantt_chart_df[["pre_script_delay","ready_delay","wms_delay", "queue_delay", "runtime", "post_script_delay",'stage_in_delay','stage_out_delay']]
    gantt_chart_delays["sum"]        = gantt_chart_delays["ready_delay"] + gantt_chart_delays["wms_delay"] \
                                        + gantt_chart_delays["queue_delay"] + gantt_chart_delays["runtime"] \
                                        + gantt_chart_delays["post_script_delay"]
    gantt_chart_delays["runtime"]   =  gantt_chart_delays["runtime"]- (gantt_chart_delays['stage_in_delay'] + gantt_chart_delays['stage_out_delay'])
    max_sum                          = gantt_chart_delays["sum"].nlargest(n=1)
    max_sum                          = max_sum.values[0]
    gantt_chart_delays["finished"]   = max_sum - gantt_chart_delays["sum"]
    
    return gantt_chart_delays, max_sum


def prepare_for_plotting(gantt_chart_delays_df, columns_names):
    jobs_data_delays_ready = gantt_chart_delays_df.to_dict('records')    
    num_jobs      = len(jobs_data_delays_ready)
    data_orders   = []
    for i in range(num_jobs):
        data_orders.append(columns_names)
    return jobs_data_delays_ready, data_orders


def plot_data(dataset, data_orders,max_sum, columns_names, save_file_name):
    # Color scheme options
    colors  = ["#ea4335","#fbbc03" ,"#33a853","#740264","black","#4cd195","#4285f4" ,"#ea8023","#5f6368"]
    colors1 = ["#4cd195","#63dbcb" ,"#ec466f","#4b4e76","#f18e8e","#f67d49"]
    colors2 = ["#4cd195", "#f1f58c","#92baed","#adf7b8","lime","#f18e8e"] 
    
    names  = columns_names
    values = np.array([[data[name] for name in order] for data,order in zip(dataset, data_orders)])
    lefts  = np.insert(np.cumsum(values, axis=1),0,0, axis=1)[:, :-1]    
    num_jobs      = len(dataset)
    
    orders  = np.array(data_orders)
    bottoms = np.arange(len(data_orders))
    
    #f, ax = plt.subplots(figsize=(1024/MY_DPI,1024/MY_DPI),dpi = MY_DPI)
    f, ax = plt.subplots(figsize=(30,30))
    plt.xlim([0,max_sum])
    plt.ylim(0,num_jobs-1)

    for name, color in zip(names, colors):
        idx   = np.where(orders == name)
        value = values[idx]
        left  = lefts[idx]
        plt.bar(x = left, height = 1.0, width = value, bottom = bottoms, 
                color = color, orientation = "horizontal", label = name)
#    Legend and ticks only for the slides or paper
#    plt.yticks(bottoms + 0.4, ["job %d" % (t) for t in bottoms])
#    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#              fancybox = True, shadow = True, ncol = 6, prop={'size': 80})
    plt.axis('off')

    #plt.savefig('{}.png'.format(save_file_name),dpi = MY_DPI,bbox_inches='tight', pad_inches=0)
    plt.savefig('{}.png'.format(save_file_name),bbox_inches='tight', pad_inches=0)
    plt.close()
    return


def create_dir_structure(img_destination_path, main_labels):
    for label in main_labels:
        path = img_destination_path + label
        try:
            os.makedirs(path)
        except:
            print("Directory structure already exists")




def main():
    # one of 3 datasets: train, test, validate
    datasets = ["test", "validation","train"]
    for dataset in datasets:
        data = pd.read_csv(SPLIT_FILES_DIR + dataset +'_data.csv')
        main_labels = data.main_label.unique()
        create_dir_structure(img_destination_path + dataset + "/", main_labels)
        for index,row in data.iterrows() :
            output_path = img_destination_path + dataset + "/" + row.main_label + "/" + row.label + "_" + row.filename.split(".")[0]
            print(output_path)
            gantt_chart = preprocess_df(row.path, cols_drop)
            gantt_chart_delays, max_sum = add_delays(gantt_chart)
            jobs_data_delays_ready, data_orders = prepare_for_plotting(gantt_chart_delays, columns_names)
            plot_data(jobs_data_delays_ready, data_orders, max_sum, columns_names,output_path)
            gc.collect()

if __name__ == "__main__":
    main()