import ast 
import json 
import crystal_math_maths
import numpy as np

with open("../Source_Data/fragment_list.json","r") as f:
    fragment_list = json.load(f)

fragment_list_reference = {}
for fragment in fragment_list:
    fragment_atoms_mass = np.array(fragment_list[fragment]["mass"])
    fragment_atoms_pos = np.array(fragment_list[fragment]["pos"])
    fragment_com = crystal_math_maths.center_of_mass(fragment_atoms_mass,fragment_atoms_pos)
    fragment_atoms_bv = fragment_atoms_pos - fragment_com

    inertia_eigenvalues, inertia_eigenvectors = crystal_math_maths.inertia(fragment_atoms_mass,
                                                                           fragment_atoms_bv)
        
    inertia_eigenvalues, inertia_eigenvectors = crystal_math_maths.sort_eigenvectors(inertia_eigenvalues, 
                                                                                     inertia_eigenvectors)
    
    inertia_eigenvectors = crystal_math_maths.ensure_right_handed_coordinate_system(inertia_eigenvectors)
    
    fragment_atoms_sfc = np.matmul(fragment_atoms_bv, 
                                    inertia_eigenvectors)
    
    fragment_atoms_sfc = np.round(fragment_atoms_sfc, decimals=4)

    fragment_list_reference[fragment] = {"smarts": fragment_list[fragment]["smarts"],
                                         "atoms": fragment_list[fragment]["atoms"],
                                         "pos": fragment_atoms_sfc.tolist(),
                                         "mass": fragment_list[fragment]["mass"],
                                         "atoms_to_align": fragment_list[fragment]["atoms_to_align"]}

# Write the reference fragment to json file
with open('../Source_Data/reference_fragment_list.json', 'w') as f:
    json.dump(fragment_list_reference, f, indent=4)  
    

