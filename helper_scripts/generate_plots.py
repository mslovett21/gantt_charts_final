import re
import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.backends.backend_pdf import PdfPages
from plotting_functions import generate_single_dist,generate_condition_wf_dist,generate_aggregate_single_dist, generate_main_type_dist, generate_agg_condition_wf_dist


def get_arguments():
    
    parser = argparse.ArgumentParser(description="Plot Gantt Charts Distributions")    
    parser.add_argument('--data_path', type=str, default='../raw_gantt_chart_data_balanced/output/',help='path to dataset ')
    parser.add_argument('--output_fname', type=str, default = '../wf_distributions_balanced.pdf' ,help='name of the output pdf file')   
    args = parser.parse_args()
    
    return args



def main():

	args = get_arguments()

	# regex pattern to find name of the workflow
	pattern = r'\-\d{3,}\S+.csv'

	#bar plt
	colors  = ['red', 'orange', 'blue', 'green', 'purple']
	pdfname = args.output_fname
	pdf     = PdfPages(pdfname)

	REL_DATA_PATH = args.data_path
	data_condition_dir = glob.glob(REL_DATA_PATH + '*')
	data_condition_dir.sort()

	all_total, wf, labels, pdf = generate_agg_condition_wf_dist(data_condition_dir, pdf, colors,pattern)
	all_total, wf, labels, pdf = generate_condition_wf_dist(data_condition_dir, pdf, colors, pattern)
	labels, wf, all_total, pdf = generate_single_dist(labels, wf, all_total, pdf)
	labels, wf, all_total, pdf = generate_aggregate_single_dist(labels, wf, all_total, pdf)
	pdf.close()



if __name__ == '__main__':
	main()

