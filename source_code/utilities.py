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