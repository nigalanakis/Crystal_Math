import json 
import re 

def convert_seconds_to_hms(seconds):
    '''
    Converts CPU time in HMS format
    
    Parameters
    ----------
    seconds : float
        The input time in seconds.
        
    Returns
    -------
    The time in HMS format.
    '''
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return int(hours), int(minutes), round(seconds, 2)

class NumpyArrayEncoder(json.JSONEncoder):
    ''' 
    Custom encoder for numpy data types 
    '''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def format_lists(json_str):
    ''' 
    Formats lists in the JSON string to remove unnecessary whitespace and newlines. 

    Parameters : str
    ----------------
        The input JSON string
    
    Returns
    -------
    The JSON string with unnecessary whitespace and newlines removed.
    '''
    pattern = re.compile(r'\[\s*((?:[^[\]]|\n)+?)\s*\]', re.DOTALL)
    return re.sub(pattern, lambda x: '[' + x.group(1).replace('\n', '').replace(' ', '') + ']', json_str)

def convert_to_json(data):
    ''' 
    Converts Python dictionary to formatted JSON string. 

    Parameters
    ----------
    data : dict
        The dictionary that needs to be converted to formatted JSON string.
    
    Returns
    -------
    formatted_json : str
        The formatted JSON string corresponding to the data dictionary.
    '''
    json_str = json.dumps(data, cls=NumpyArrayEncoder, indent=4)
    formatted_json = format_lists(json_str)
    return formatted_json