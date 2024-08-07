import json
from datetime import datetime
from time import process_time as timer   

from csd_operations import cluster_refcode_families
from csd_operations import get_refcode_families
from csd_operations import get_unique_structures
from generate_molecule_fragments import create_reference_fragments
from get_structure_data import get_structure_data, get_structure_filter_data
from utilities import convert_seconds_to_hms

def main(input_file):
    # Load execution parameters 
    with open('input_files/' + input_file) as f:
        input_parameters = json.load(f)
    
    # Get the refcode families
    if input_parameters["get_refcode_families"]:
        print('Getting the CSD refcode families and the structures in each family.')
        get_refcode_families(input_parameters)
        
    # Cluster refcode families based on structure similarity.
    if input_parameters["cluster_refcode_families"]:
        print('Filter structures based on the user defined criteria and clustering refcode families members based on packing similarity.')
        cluster_refcode_families(input_parameters)
        
    # Get unique structures
    if input_parameters["get_unique_structures"]:
        print('Getting unique structures.')
        get_unique_structures(input_parameters)
        
    # Get structure data
    if input_parameters["get_structure_data"]:
        print('Getting structure data.')
        get_structure_data(input_parameters)
        
    # Get structure filter data
    if input_parameters["get_structure_filter_data"]:
        print('Getting structure filter data.')
        get_structure_filter_data(input_parameters)

if __name__ == "__main__":
    input_file = "input_data_extraction.txt"

    now = datetime.now()    
    print('#' * 80)
    print('Crystal Math')
    print('A Mathematical and Geometrical Crystal Structure Analyis Protocol')
    print('-' * 80)
    print('Nikos Galanakis')
    print('Research Scientist')
    print('The Tuckerman Group')
    print('New York University')
    print('ng1807@nyu.edu')
    print('=' * 80)
    print("Process started at ", now.strftime("%Y-%m-%d %H:%M:%S"))
    print('-' * 80)
    
    start = timer()
    main(input_file)
    
    cpu_time = timer() - start
    hours, minutes, seconds = convert_seconds_to_hms(cpu_time)
    now = datetime.now()
    print("Process completed at ", now.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Total computation time: {hours}h {minutes}m {seconds:.2f}s")