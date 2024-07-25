import copy 
import json 
import numpy as np
from collections import OrderedDict
from space_group_operations import apply_symmetry_operations
from structure_operations import get_lattice_vectors

def get_user_variables(input_parameters,variables):
    '''
    Gets the variables for the analysis/plotting
    
    Parameters
    ----------    
    input_parameters : dict
        The user defined input file
    variables : dict
        A dictionary with the available variables
        
    Returns
    -------
    A list with the unique variables for the analysis/plotting
    '''
    # Get the variables for the analysis
    user_variables = []
    if input_parameters['histograms_options']['variables'] != 'all':
        user_variables.extend([var[0][0] for var in input_parameters['histograms_options']['variables']])
    else:
        for key in variables:
            for var in variables[key]:
                user_variables.append(var)
    if input_parameters['2D_scatter_plots_options']['variables'] != 'all':
        user_variables.extend([var[0][0] for var in input_parameters['2D_scatter_plots_options']['variables']])
        user_variables.extend([var[0][1] for var in input_parameters['2D_scatter_plots_options']['variables']])
    else:
        for key in variables:
            for var in variables[key]:
                user_variables.append(var)
    if input_parameters['3D_scatter_plots_options']['variables'] != 'all':
        user_variables.extend([var[0][0] for var in input_parameters['3D_scatter_plots_options']['variables']])
        user_variables.extend([var[0][1] for var in input_parameters['3D_scatter_plots_options']['variables']])
        user_variables.extend([var[0][2] for var in input_parameters['3D_scatter_plots_options']['variables']])
    else:
        for key in variables:
            for var in variables[key]:
                user_variables.append(var)
        
    return sorted(list(set(user_variables)))

def get_value(data, path):
    '''
    Gets value for a specific variable

    Parameters
    ----------
    data : dict
        A dictionary with the structure data.
    path : str
        The path to read the variable value from the structure data dictionary.

    Returns
    -------
    data : float, str or bool
        The value for the specific variable in the structure.
    '''
    for key in path:
        data = data[key]
    return data

def get_analysis_structures_list(input_parameters):
    # Get the structures list based on the user defined filters
    structures_list = {}
    structures_filter_data_filename = input_parameters['data_directory'] + input_parameters['data_prefix'] + '_structures_filter_data.json'

    filter_groups = {
        'single': {
            'target_space_groups': 'space_group',
            'target_z_crystal_values': 'z_crystal',
            'target_z_prime_values': 'z_prime',
            'target_species': 'species'
        },
        'single_combinations': {
            'target_structure_fragments': 'fragments',
            'target_contact_central_fragments': 'contact_central_fragments'
        },
        'multiple_combinations': {
            'target_contact_pairs': 'contact_pairs',
            'target_contact_fragment_pairs': 'contact_fragment_pairs'
        }
    }
            
    with open(structures_filter_data_filename) as f:
        structures_filter_data = json.load(f)
        
        for structure, values in zip(structures_filter_data.keys(),structures_filter_data.values()):
            accept_structure = True

            if input_parameters['target_families'] != None and structure[:6] not in input_parameters['target_families']:
                accept_structure = False

            if input_parameters['target_structures'] != None and structure not in input_parameters['target_structures']:
                accept_structure = False

            for filter, property in zip(filter_groups['single'].keys(),filter_groups['single'].values()):
                if input_parameters[filter] != None and values[property] not in input_parameters[filter]:
                    accept_structure = False

            for filter, property in zip(filter_groups['single_combinations'].keys(),filter_groups['single_combinations'].values()):
                if input_parameters[filter] != None:
                    if input_parameters[filter][1] == 'or':
                        if not any(item in set(values[property]) for item in input_parameters[filter][0]):
                            accept_structure = False
                    elif input_parameters[filter][1] == 'and':
                        if not set(input_parameters[filter][0]).issubset(set(values[property])):
                            accept_structure = False

            for filter, property in zip(filter_groups['multiple_combinations'].keys(),filter_groups['multiple_combinations'].values()):
                if input_parameters[filter] != None:
                    if input_parameters[filter][1] == 'or':
                        if not any(tuple(item) in set(tuple(x) for x in values[property]) for item in input_parameters[filter][0]):
                            accept_structure = False
                    elif input_parameters[filter][1] == 'and':
                        if not set(tuple(x) for x in input_parameters[filter][0]).issubset(set(tuple(x) for x in values[property])):
                            accept_structure = False

            if not accept_structure:
                continue 
            
            space_group = structures_filter_data[structure]['space_group']
            if space_group != 'R-3':
                if space_group not in structures_list:
                    structures_list[space_group] = []
                structures_list[space_group].append(structure)
    return structures_list
    
def get_analysis_data(input_parameters,variables):
    '''
    Gets the data for the plots.

    Parameters
    ----------
    input_parameters : dict
        A dictionary with the user defined input parameters.
    variables : dict
        A dictionary with the available variables and their properties.

    Returns
    -------
    data : dict
        A dictionary with the required data to create the user defined plots.

    '''
    
    # Get the variables for the analysis
    user_variables = get_user_variables(input_parameters,variables)
    
    # Get the structures list for the analysis based on the used defined filters
    structures_list = get_analysis_structures_list(input_parameters)

    # Set the structures folder
    structure_files_folder = input_parameters['data_directory'] + input_parameters['data_prefix'] + '_structures/'

    # Get the user variables families
    variable_families = [variables[var]['family'] for var in user_variables]
    variable_families = sorted(list(set(variable_families)))

    # Initialize  the data dictionary for the analysis
    data = {key: {space_group: {} for space_group in structures_list} for key in variable_families}

    # Set the data filter dependencies (variables that should be added for filtering data)
    filter_dependencies = {
        'structure': ['z_crystal','z_prime'],
        'fragment': ['z_crystal','z_prime','fragment'],
        'contact': ['z_crystal','z_prime','cc_central_atom_fragment','cc_contact_atom_fragment','cc_central_atom_species','cc_contact_atom_species','cc_type','cc_is_in_los'],
        'fragment_atom': ['z_crystal','z_prime','fragment','fragment_atom_species'],
        'contact_atom': ['z_crystal','z_prime','cc_central_atom_fragment','cc_contact_atom_fragment','cc_central_atom_species','cc_contact_atom_species','cc_type','cc_is_in_los','cc_length']}

    # Add variables to the data ditionary for data filter dependencies
    for variable_family in variable_families:
        for space_group in structures_list:
            for denepdency in filter_dependencies[variable_family]:
                data[variable_family][space_group][denepdency] = []

    # Add user variables to data dictionary
    for var in user_variables:
        for space_group in structures_list:
            data[variables[var]['family']][space_group][var] = []

    # Check if additional positional variables should be added to calculate coordinate transformations from fractional to cartesian and via versa.
    variable_groups = {}
    for var in variables:
        variable_group = variables[var]['position_symmetry'][3]
        if str(variable_group) not in variable_groups:
            variable_groups[str(variable_group)] = []
        
        variable_groups[str(variable_group)].append(var)

    for variable_family in data:
        for space_group in structures_list:
            add_zero = False
            family_variables = [key for key in data[variable_family][space_group]]
            for var in family_variables:
                variable_group = variables[var]['position_symmetry'][3]
                if variable_group > 0 and var[-2:] in ['_x','_y','_z','_u','_v','_w']:
                    add_zero = True
                    for group_var in variable_groups[str(variable_group)]:
                        if group_var not in data[variable_family][space_group]:
                            data[variable_family][space_group][group_var] = [] 
            if add_zero:
                for var in variable_groups['0']:
                    data[variable_family][space_group][var] = [] 

    # Get data from indivirdual structure files grouped by space group
    for space_group in structures_list:
        for structure in structures_list[space_group]:
            with open(structure_files_folder + structure + '.json') as f:
                # Read the current structure data
                structure_data = json.load(f)

            # Add structure data to the data dictionary for the analysis
            for variable_family in variable_families:
                for var in data[variable_family][space_group]:
                    
                    if variable_family == 'structure': 
                        path = copy.deepcopy(variables[var]['path']) 
                        value = get_value(structure_data, path)
                        data[variable_family][space_group][var].append(value)
                    
                    if variable_family == 'fragment': 
                        if variables[var]['family'] == 'fragment':
                            for fragment_key in structure_data['fragments']:
                                path = copy.deepcopy(variables[var]['path'])
                                path[1] = fragment_key
                                value = get_value(structure_data, path)
                                data[variable_family][space_group][var].append(value)
                        if variables[var]['family'] == 'structure':
                            for fragment_key in structure_data['fragments']:
                                path = copy.deepcopy(variables[var]['path'])
                                value = get_value(structure_data, path)
                                data[variable_family][space_group][var].append(value)
                        
                    if variable_family == 'fragment_atom': 
                        if variables[var]['family'] == 'fragment_atom':
                            for fragment_key in structure_data['fragments']:
                                path = copy.deepcopy(variables[var]['path'])
                                path[1] = fragment_key
                                for atom_key in structure_data['fragments'][fragment_key]['atoms']:
                                    path[3] = atom_key
                                    value = get_value(structure_data, path)
                                    data[variable_family][space_group][var].append(value)
                        if variables[var]['family'] == 'fragment':
                            for fragment_key in structure_data['fragments']:
                                for atom_key in structure_data['fragments'][fragment_key]['atoms']:
                                    path = copy.deepcopy(variables[var]['path'])
                                    path[1] = fragment_key
                                    value = get_value(structure_data, path)
                                    data[variable_family][space_group][var].append(value)
                        if variables[var]['family'] == 'structure':
                            for fragment_key in structure_data['fragments']:
                                for atom_key in structure_data['fragments'][fragment_key]['atoms']:
                                    path = copy.deepcopy(variables[var]['path'])
                                    value = get_value(structure_data, path)
                                    data[variable_family][space_group][var].append(value)
                                
                    if variable_family == 'contact':
                        if variables[var]['family'] == 'contact':
                            for pair_key in structure_data['crystal']['close_contacts']:
                                path = copy.deepcopy(variables[var]['path']) 
                                path[2] = pair_key
                                value = get_value(structure_data, path)
                                data[variable_family][space_group][var].append(value)  
                        if variables[var]['family'] == 'contact_atom' and var in ['cc_central_atom_fragment','cc_contact_atom_fragment','cc_central_atom_species','cc_contact_atom_species']:
                            for pair_key in structure_data['crystal']['close_contacts']:
                                path = copy.deepcopy(variables[var]['path']) 
                                path[2] = pair_key
                                path[3] = var[:15]
                                value = get_value(structure_data, path)
                                data[variable_family][space_group][var].append(value)
                        if variables[var]['family'] == 'structure':
                            for pair_key in structure_data['crystal']['close_contacts']:
                                path = copy.deepcopy(variables[var]['path'])
                                value = get_value(structure_data, path)
                                data[variable_family][space_group][var].append(value)
                                
                    if variable_family == 'contact_atom':
                        if variables[var]['family'] == 'contact_atom':
                            for pair_key in structure_data['crystal']['close_contacts']:
                                path = copy.deepcopy(variables[var]['path']) 
                                path[2] = pair_key
                                path[3] = var[:15]
                                value = get_value(structure_data, path)
                                data[variable_family][space_group][var].append(value) 
                        if variables[var]['family'] == 'contact':
                            for pair_key in structure_data['crystal']['close_contacts']:
                                path = copy.deepcopy(variables[var]['path']) 
                                path[2] = pair_key
                                value = get_value(structure_data, path)
                                data[variable_family][space_group][var].append(value)  
                        if variables[var]['family'] == 'structure':
                            for pair_key in structure_data['crystal']['close_contacts']:
                                path = copy.deepcopy(variables[var]['path'])
                                value = get_value(structure_data, path)
                                data[variable_family][space_group][var].append(value)

    # Apply symmetry operations to atomic coordinates
    # Get the space group properties
    with open('../source_data/space_group_properties.json') as f:
        space_group_properties = json.load(f)
        
    for variable_family in data:
        for space_group in data[variable_family]:
            # Sort data based on variable name for the correct application of symmetry operations 
            data[variable_family][space_group] = OrderedDict(sorted(data[variable_family][space_group].items()))

            # Load symmetry operations for the current space group
            symmetry_operations = space_group_properties[space_group]['symmetry_operations']

            # Get the groups of variables
            groups = []
            for var in data[variable_family][space_group].keys():
                group = variables[var]['position_symmetry']
                if group[3] > 0 and group not in groups:
                    groups.append(group)

            # Get the symmetry groups
            symmetry_groups = []
            for group in groups:
                group_variables = [var for var in variables if variables[var]['position_symmetry'][3] == group[3]]
                symmetry_groups.append([group[0],group[1],group_variables])

            # Get the coordinates for the symemtric atoms
            symmetric_variables = []
            for rotation, translation, group_variables in symmetry_groups:
                symmetric_variables.extend([var for var in group_variables])
                if group_variables[0][-2:] == '_u':
                    fractional_positions = np.transpose([data[variable_family][space_group][group_variables[0]],
                                                          data[variable_family][space_group][group_variables[1]],
                                                          data[variable_family][space_group][group_variables[2]]])
                    
                elif group_variables[0][-2:] == '_x':
                    cartesian_positions = np.transpose([data[variable_family][space_group][group_variables[0]],
                                                        data[variable_family][space_group][group_variables[1]],
                                                        data[variable_family][space_group][group_variables[2]]])
                    cell_parameters = np.transpose([data[variable_family][space_group]['cell_length_a'],
                                                    data[variable_family][space_group]['cell_length_b'],
                                                    data[variable_family][space_group]['cell_length_c'],
                                                    data[variable_family][space_group]['cell_angle_alpha'],
                                                    data[variable_family][space_group]['cell_angle_beta'],
                                                    data[variable_family][space_group]['cell_angle_gamma'],
                                                    data[variable_family][space_group]['cell_volume']])
                    
                    fractional_positions = []
                    lattice_vectors = []
                    for (x, y, z), (a, b, c, alpha, beta, gamma, omega) in zip(cartesian_positions,cell_parameters):
                        lattice_vectors.append(get_lattice_vectors(np.array([a,b,c]),np.array([alpha,beta,gamma]),omega,inverse=False))
                        inverse_lattice_vectors = get_lattice_vectors(np.array([a,b,c]),np.array([alpha,beta,gamma]),omega,inverse=True)
                        fractional_positions.append(np.dot([x,y,z],inverse_lattice_vectors).tolist())
                
                for op in symmetry_operations[1:]:
                    if group_variables[0][-2:] == '_u':
                        symmetric_positions = apply_symmetry_operations(fractional_positions, op, translation)  
                    if group_variables[0][-2:] == '_x':
                        symmetric_positions = []
                        for pos, vec in zip(fractional_positions,lattice_vectors):
                            symmetric_positions.append(np.dot(apply_symmetry_operations([pos], op, translation), vec)[0])

                    for x, y, z in symmetric_positions:
                        data[variable_family][space_group][group_variables[0]].append(x)
                        data[variable_family][space_group][group_variables[1]].append(y)
                        data[variable_family][space_group][group_variables[2]].append(z)
            
            # Extend data for symmetric positions
            if symmetric_variables != []:
                for var in data[variable_family][space_group]:
                    if var not in symmetric_variables:
                        extend_data = copy.deepcopy(data[variable_family][space_group][var])
                        for i in range(len(symmetry_operations) - 1):
                            data[variable_family][space_group][var].extend(extend_data)
                            
            # Move fractional coordinates of atoms in unit cell if necessary
            for var in data[variable_family][space_group]:
                if variables[var]['position_symmetry'][2]:
                    data[variable_family][space_group][var] = [x % 1 for x in data[variable_family][space_group][var]]

    return data