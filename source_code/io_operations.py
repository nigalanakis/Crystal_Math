import os 

def check_for_file(save_dir,filename):
    """
    Check if a file exists and if it does, ask the user whether to overwrite it or not.

    Parameters
    ----------
    filename : str
        The name of the file
    
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

            
    
    
    
