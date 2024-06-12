import os 

def check_for_file(save_dir,filename):
    """
    Check if a file exists and if it does, ask the user whether to overwrite it or not.

    Parameters
    ----------
    filename : str
        The name of the file.
    
    Returns
    -------
    file object
    """
    if os.path.exists(save_dir + filename):
        # If the file exists, ask for confirmation to overwrite
        user_input = input(f'File {filename} already exists. Do you want to overwrite it? (yes/no): ').lower()
        if user_input not in ['yes','y','YES','Y']:
            print(f'\tWARNING! Exiting without overwriting the file.\n\tNo data will be writen in file: {filename}')
            return None  # or manage the scenario where the user doesn't want to overwrite

    # If the file doesn't exist or if overwrite is confirmed, open and return the file object
    try:
        file_object = open(save_dir + filename, 'w')  # Open the file with writing mode, which will also create it if it doesn't exist
        return file_object
    except Exception as e:
        print(f'An error occurred: {e}')
        return None 

def write_structure_data_file(db_structures_folder,structure_crystal,structure_data):
    """
    Writes structure data to a json format file.
    
    Parameters
    ----------
    db_structures_folder : str
        The folder where the data for each structure will be stored.
    structure_crystal : dict
        A dictionary with the structure crystal data.
    structure_data : dict
        A dictionary with the structure data.
    
    Returns
    -------
    """

    structure_data_file = db_structures_folder + structure_crystal["str_id"] + ".json"
    with open(structure_data_file,"w") as f:
        f.write(structure_data)
        
    return

def write_structures_filter_data(input_parameters,structures_filter_data):
    """ 
    Writes compact structure data for the filtering step.
    
    Parameters
    ----------
    input parameters : dict
        A dictionary with the user defined input data.
    structures_filter_data : dict
        A dictionary with the compact structure data.
        
    Returns
    -------
    None
    
    """
    # Set the file name
    db_folder = "../csd_db_analysis/db_data/"
    prefix = input_parameters["data_prefix"] 
    structures_filter_data_file = check_for_file(db_folder, prefix + "_structures_filter_data.json")
    
    # Write data and close file
    structures_filter_data_file.write(structures_filter_data)
    structures_filter_data_file.close()
    
    return
    
    