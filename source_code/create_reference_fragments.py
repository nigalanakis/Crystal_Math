import json 
import numpy as np

from maths import calculate_inertia
from maths import center_of_mass
from maths import ensure_right_handed_coordinate_system
from maths import sort_eigenvectors

def create_reference_fragments():
    """ 
    Converts the input fragment list into a space fixed list of fragments.
    
    Parameters
    ----------
    
    Returns
    -------
    reference_fragment_list : dict
        A dictionary with the space fixed reference fragments.
    """
    with open("../source_data/fragment_list.json","r") as f:
        fragment_list = json.load(f)
    
    reference_fragment_list = {}
    for fragment in fragment_list:
        fragment_atoms_mass = np.array(fragment_list[fragment]["mass"])
        fragment_atoms_pos = np.array(fragment_list[fragment]["coordinates"])
        fragment_com = center_of_mass(fragment_atoms_mass,fragment_atoms_pos)
        fragment_atoms_bv = fragment_atoms_pos - fragment_com
    
        inertia_eigenvalues, inertia_eigenvectors = calculate_inertia(fragment_atoms_mass,
                                                                      fragment_atoms_bv)
            
        inertia_eigenvalues, inertia_eigenvectors = sort_eigenvectors(inertia_eigenvalues, 
                                                                      inertia_eigenvectors)
        
        inertia_eigenvectors = ensure_right_handed_coordinate_system(inertia_eigenvectors)
        
        fragment_atoms_sfc = np.matmul(fragment_atoms_bv, 
                                        inertia_eigenvectors)
        
        fragment_atoms_sfc = np.round(fragment_atoms_sfc, decimals=4)
    
        reference_fragment_list[fragment] = {"smarts": fragment_list[fragment]["smarts"],
                                             "species": fragment_list[fragment]["species"],
                                             "coordinates_sf": fragment_atoms_sfc.tolist(),
                                             "mass": fragment_list[fragment]["mass"],
                                             "atoms_to_align": fragment_list[fragment]["atoms_to_align"]}
    
    # Write the reference fragment to json file
    with open('../source_data/reference_fragment_list.json', 'w') as f:
        json.dump(reference_fragment_list, f, indent=4)  
        
    return reference_fragment_list
    

