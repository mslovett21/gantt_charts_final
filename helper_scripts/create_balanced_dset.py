import os
import re
import glob
import random
from os.path import exists


random.seed(10)

REL_DATA_PATH = '../raw_gantt_chart_data_balanced/output/'
data_condition_dir = glob.glob(REL_DATA_PATH + '*')
data_condition_dir.sort()
pattern = r'\-\d{3,}\S+.csv'




def delete_files(wf_name, condition_type, num_delete):
    print(condition_type)
    
    type_path = REL_DATA_PATH + condition_type
    all_type_files = glob.glob(type_path + '/*')
    wf_type_list   = [ x for x in all_type_files if wf_name in x]
    random.shuffle(wf_type_list)
    files_to_delete = random.sample(wf_type_list, num_delete)
    i = 0
    
    for file in files_to_delete:
        if exists(file):
            os.remove(file)
            i = i+1
    print("Removed {} out of {}".format(i,num_delete))
    
    return



def delete_other_wf_types(condition_type, wf_type_list, num_delete):
    
    for wf_name in wf_type_list:
        files_to_delete_hdd70 = delete_files(wf_name, condition_type, num_delete)
        
    return



def main():

	wf_type_list   = ['1000genome','nowcast-clustering-16-casa_nowcast-wf','nowcast-clustering-8-casa_nowcast-wf',
            'wind-clustering-casa_wind_wf', 'wind-noclustering-casa_wind_wf']

	hdd_subtypes = ['hdd_50', 'hdd_60', 'hdd_70', 'hdd_80', 'hdd_90', 'hdd_100' ]
	num_delete   = 15


	for condition_type in hdd_subtypes:
	    delete_other_wf_types(condition_type, wf_type_list, num_delete)

	wf_name    = '1000genome'

	condition_type = 'hdd_100'
	num_delete = 50 # 50 + 15 = 65 overall
	files_to_delete_hdd100= delete_files(wf_name, condition_type, num_delete)



	condition_type = 'hdd_50'
	num_delete = 75 # 75+15 = 90 overall
	files_to_delete_hdd50 = delete_files(wf_name, condition_type, num_delete)

	return








if __name__ == '__main__':
	main()

