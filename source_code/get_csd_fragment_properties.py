import numpy as np 
from center_of_mass import center_of_mass
        
def get_csd_fragment_properties(fragment,molecule):
    fragment_properties = {}
    fragment_properties["atoms"] = fragment["atoms"] # Get the atom list for the fragment
    fragment_properties["species"] = fragment["species"] # Get the atom list for the fragment
    fragment_properties["atoms_mass"] = molecule["atoms_mass"][fragment_properties["atoms"]] # Get the mass of the atoms in the fragment
    fragment_properties['atom_coordinates_c'] = molecule["atoms_coordinates_c"][fragment_properties["atoms"]] # Get the physical positions of the atoms in the fragment
    fragment_properties["atoms_coordinates_sf"] = np.array(fragment["coordinates_sf"]) # Get the reference positions of the atoms for the fragment in the body fixed frame
    fragment_properties["coordinates_c"] = center_of_mass(fragment_properties["atoms_mass"],fragment_properties["atoms_coordinates_c"]) # Calculate the center of mass for the fragment
    fragment_properties["atoms_bond_vectors_c"] = fragment_properties['atom_coordinates_c'] - fragment_properties['coordinates_c'] # Calculate the bond vectors for the fragment
    
    return fragment_properties