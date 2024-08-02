import sys 

# Define a function to check if the given variables are boolean
def check_boolean_variables(variables):
    for var_name, value in variables.items():
        if not isinstance(value, bool):
            print(f"Error: The variable '{var_name}' must be a boolean (True or False).")
            sys.exit(1)

# Define a function to check if the given variables are integers
def check_integer_variables(variables):
    for var_name, value in variables.items():
        if not isinstance(value, int):
            print(f"Error: The variable '{var_name}' must be an integer (True or False).")
            sys.exit(1)

# Define functions to check if the single value variables get values from their respective lists
def check_single_value_variables(variables):
    for var_name, values in variables.items():
        if values[0] not in values[1]:
            print(f"Error: The variable '{var_name}' has an invalid value '{values[0]}'. Allowed values are {values[1]}.")
            sys.exit(1)

# Define functions to check if the list variables get values from their respective lists
def check_list_variables(variables):
    for var_name, values in variables.items():
        if any(value not in values[1] for value in values[0]):
            print(f"Error: The variable '{var_name}' has an invalid value '{values[0]}'. Allowed values are {values[1]}.")
            sys.exit(1)

def check_input_parameters(data_analysis,data_extraction,extraction_actions,extraction_filters,analysis_actions,topological_properties):
    # Define the check for the boolean variables
    boolean_variables = {
        'data_analysis': data_analysis,
        'data_extraction': data_extraction,
        **extraction_actions,
        'center_molecule': extraction_filters['center_molecule'],
        'add_full_component': extraction_filters['add_full_component'],
        **analysis_actions
    }
    
    # Define the dictionary mapping integer variables to their available values
    integer_variables = {
        'proposed_vectors_n_max': topological_properties['proposed_vectors_n_max']
    }
        
    # Define the dictionary mapping single value variables to their available values
    single_value_variables = {
        'unique_structures_clustering_method': [extraction_filters['unique_structures_clustering_method'], ['energy', 'vdWFV']],
        'structure_list': [extraction_filters['structure_list'][0], ['csd-all', 'csd-unique', 'cif']]
    }
    
    # Define the dictionary mapping list variables to their available values
    list_variables = {
        'crystal_type': [extraction_filters['crystal_type'], ['homomolecular', 'co-crystal', 'hydrate']]
    }
    
    check_boolean_variables(boolean_variables)
    check_integer_variables(integer_variables)
    check_single_value_variables(single_value_variables)
    check_list_variables(list_variables)