import ast 
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from time import process_time as timer  

import crystal_math_visualization 
import crystal_math_maths


# File paths to data directory and plots directory
mwd = '../CSD_DB_Analysis/'
data_dir = mwd + 'DB_Data/'
plots_dir = mwd + 'Plots/'
contacts_plots_dir = plots_dir + 'Contacts/'

contacts_data_file = data_dir + 'contacts_data.txt'
structure_data_file = data_dir + 'structure_data.txt'

os.makedirs(plots_dir, exist_ok=True)
os.makedirs(contacts_plots_dir, exist_ok=True)

def main(input_file):
    # Set latex rendering for the plots
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern Roman'
    # Optionally, set the font size
    plt.rcParams['font.size'] = 10
    
    with open(input_file) as f:
        data = f.read()
    plot_parameters = ast.literal_eval(data)
    
    ''' Create histograms for the close contacts length '''
    max_length = plot_parameters['contacts_max_length']
    bin_width = plot_parameters['contacts_bin_width']
    bins = np.arange(0, max_length + bin_width, bin_width)
    crystal_math_visualization.create_close_contacts_histograms(plot_parameters['structure_data_file'],
                                                                plot_parameters['contacts_data_file'],
                                                                plot_parameters['contacts_plots_dir'],
                                                                bins)
    
    ''' Create histograms for the hydrogen bonds lengths and angles '''
    max_length = plot_parameters['h_bonds_max_length']
    bin_width = plot_parameters['h_bonds_bin_width']
    bins = np.arange(0, max_length + bin_width, bin_width)
    crystal_math_visualization.create_h_bonds_histograms(plot_parameters['structure_data_file'],
                                                          plot_parameters['h_bonds_data_file'],
                                                          plot_parameters['h_bonds_plots_dir'],
                                                          0,
                                                          bins)
    
    max_length = plot_parameters['h_bonds_da_max_length']
    bin_width = plot_parameters['h_bonds_bin_width']
    bins = np.arange(0, max_length + bin_width, bin_width)
    crystal_math_visualization.create_h_bonds_histograms(plot_parameters['structure_data_file'],
                                                          plot_parameters['h_bonds_data_file'],
                                                          plot_parameters['h_bonds_plots_dir'],
                                                          1,
                                                          bins)
    
    min_angle = plot_parameters['h_bonds_min_angle']
    max_angle = plot_parameters['h_bonds_max_angle']
    bin_width = plot_parameters['h_bonds_angle_bin_width']
    bins = np.arange(min_angle, max_angle + bin_width, bin_width)
    crystal_math_visualization.create_h_bonds_histograms(plot_parameters['structure_data_file'],
                                                         plot_parameters['h_bonds_data_file'],
                                                         plot_parameters['h_bonds_plots_dir'],
                                                         2,
                                                         bins)
    
    
if __name__ == "__main__":
    input_file = "input_files/input_csd_data_analysis.txt"
    
    now = datetime.now()
    print("Process started at ", now.strftime("%Y-%m-%d %H:%M:%S"))

    start = timer()
    main(input_file)
    
    cpu_time = timer() - start
    hours, minutes, seconds = crystal_math_maths.convert_seconds(cpu_time)
    now = datetime.now()
    print("Process completed at ", now.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Total computation time: {hours}h {minutes}m {seconds:.2f}s")
        
    
        
