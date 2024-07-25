import numpy as np

def parse_symmetry_operation(op):
    '''
    Parse a symmetry operation string into a transformation function.
    '''
    # Replace fractional coordinate symbols with array references
    op = op.replace('x', 'coord[0]').replace('y', 'coord[1]').replace('z', 'coord[2]')
    
    # Create a lambda function for the operation
    return lambda coord: np.array(eval(op))

def apply_symmetry_operations(fractional_positions,symmetry_operation,translation=True):
    '''
    Apply symmetry operations to the fractional coordinates of a molecule and return a list for each operation.
    
    Parameters
    ----------
    fractional_positions : numpy.ndarray
        The fractional coordinates of the atoms in the reference molecule.
    symmetry_operation : str
        The symmetry operation for the symmetric molecule
        
    Returns
    -------
    symmetric_positions : numpy.ndarray
        The fractional coordinates of the atoms in the symmetric molecule
    '''
    # Parse the symmetry operations
    if not translation:
        for translation_coefficient in ["0.25","0.5","0.75","1.0","1/3","2/3","5/6"]:
            symmetry_operation = symmetry_operation.replace(translation_coefficient,"0.0")

    transformation = parse_symmetry_operation(symmetry_operation)

    # Apply transformations to each fractional position
    symmetric_positions =  np.array([transformation(pos) for pos in fractional_positions])
    
    return symmetric_positions
