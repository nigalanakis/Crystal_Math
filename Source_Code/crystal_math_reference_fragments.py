import ast 
import json 
import maths
import numpy as np

with open("../Source_Data/fragment_list.txt") as f:
    data = f.read()
    
fragment_list = ast.literal_eval(data)

fragment_list_reference = {}
for fragment in fragment_list:
    fragment_atoms_mass = np.array(fragment_list[fragment]["mass"])
    fragment_atoms_pos = np.array(fragment_list[fragment]["pos"])
    fragment_com = maths.center_of_mass(fragment_atoms_mass,fragment_atoms_pos)
    fragment_atoms_bv = fragment_atoms_pos - fragment_com

    inertia_eigenvalues, inertia_eigenvectors = maths.inertia(fragment_atoms_mass,
                                                              fragment_atoms_bv)
        
    inertia_eigenvalues, inertia_eigenvectors = maths.sort_eigenvectors(inertia_eigenvalues, 
                                                                        inertia_eigenvectors)
    
    inertia_eigenvectors = maths.ensure_right_handed_coordinate_system(inertia_eigenvectors)
    
    fragment_atoms_sfc = np.matmul(fragment_atoms_bv, 
                                    inertia_eigenvectors)
    
    fragment_atoms_sfc = np.round(fragment_atoms_sfc, decimals=4)

    fragment_list_reference[fragment] = {"smarts": fragment_list[fragment]["smarts"],
                                         "pos": fragment_atoms_sfc.tolist(),
                                         "atoms_to_align": fragment_list[fragment]["atoms_to_align"]}

file_path = "../Source_Data/reference_fragment_list.txt"

# Open the file for writing
with open(file_path, 'w') as file:
    # Iterate through the dictionary items
    file.write('{\n')
    for main_key, values in fragment_list_reference.items():
        file.write(f'"{main_key}":' + '{ \n')
        for key, value in values.items():
            # Write the key and value to the file with a new line only after the key
            if isinstance(value, str):
                file.write(f'\t"{key}": "{value}",' + '\n')
            else:
                file.write(f'\t"{key}": {value},' + '\n')
        file.write('\t},\n')
    file.write('}\n')
    
    

