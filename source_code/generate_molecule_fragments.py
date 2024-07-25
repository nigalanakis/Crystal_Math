import ast 
import json 
import numpy as np

from maths import calculate_inertia
from maths import center_of_mass
from maths import ensure_right_handed_coordinate_system
from maths import sort_eigenvectors
from structure_operations import get_atoms_from_formula

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
        
    return 

def get_reference_fragment_list():
    '''
    Returns the reference fragment list.
    '''    
    with open('../source_data/reference_fragment_list.json','r') as f:
        reference_fragment_list = json.load(f)
    return reference_fragment_list

def get_molecule_fragments(input_fragments,reference_fragment_list):
    '''
    Returns the fragments for the reference molecule.
    
    Parameters
    ----------
    input_fragments : list
        A list with the fragments for the compound.
    reference_fragment_list : dict
        A dictionary with the reference fragments used to build the molecules.
        
    Returns
    -------
    int
        The number of fragments.
    molecule_fragments : dict
        A dictionary with the properties of the fragments.
            
    '''
    molecule_fragments = {}
    fragment_count = {}  # Keep track of how many times each fragment has appeared
      
    for fragment in input_fragments:
        if fragment in fragment_count:
            # If the fragment has appeared before, increment its count
            fragment_count[fragment] += 1
            # Use the fragment name and its count to create a unique key
            unique_key = f"{fragment}_{fragment_count[fragment]}"
        else:
            # If it's the first time the fragment appears, initialize its count
            fragment_count[fragment] = 1
            unique_key = fragment
            
        # Use the unique key to store the fragment in the molecule_fragments dictionary
        molecule_fragments[unique_key] = reference_fragment_list[fragment]
    
    return len(molecule_fragments), molecule_fragments

def calculate_molecular_volume(formula,compound_rings,atomic_properties):
    '''
    Calculates the molecular vdW volume
        J. Org. Chem. 2003, 68, 19, 7368–7373

    Parameters
    ----------
    formula : str 
        The molecular formula.
    compound_rings : dictionary
        A dictionary with the number of aromatic and aliphatic rings.
    atomic_properties : dict
        A dictionary containing the atomic properties.

    Returns
    -------
    molecular_volume : float
        The molecular vdW volume.
    '''
    # Set the number of aromatic and aliphatic rings from the dictionary 
    n_rings = [compound_rings['aromatic'], compound_rings['aliphatic']]
    
    # Get the count of atoms for each species 
    species_counts, n_atoms, _ = get_atoms_from_formula(formula)

    # Calculate the number of bonds
    n_bonds = n_atoms - 1 + n_rings[0] + n_rings[1]

    # Calculate total atomic vdW volume
    atomic_vdW_volume = np.sum([species_counts[key] * (4.0 * np.pi * atomic_properties[key]['van_der_waals_radius']**3 / 3) for key in species_counts])

    # Calculate the molecular volume
    molecular_volume = atomic_vdW_volume - 5.92 * n_bonds - 14.7 * n_rings[0] - 3.8 * n_rings[1]

    return molecular_volume

def calculate_molecular_weight(formula,atomic_properties):
    '''
    Calculates the molecular weight from formula

    Parameters
    ----------
    formula : str
        The molecular formula.
    atomic_properties : dict
        A dictionary containing the atomic properties.

    Returns
    -------
    molecular_weight : float
        The molecular weight
    '''
    # Get the count of atoms for each species 
    species_counts, n_atoms, _ = get_atoms_from_formula(formula)

    # Calculate the molecular weight
    molecular_weight = np.sum([species_counts[atom]*atomic_properties[atom]['atomic_mass'] for atom in species_counts])

    return molecular_weight

def generate_fragments(input_parameters,atomic_properties):
    '''
    Reads input data from input files.

    Parameters
    ----------
    input_parameters : dict
        A dictionary containing the input parameters.

    Returns
    -------
    n_fragments : int
        The number of fragments in the molecule.
    fragments : dict
        A dictionary with the fragment properties.
    reference_molecule : dict
        A dictionary containing the reference molecule properties.
    '''
    # Read the reference fragment list 
    reference_fragment_list = get_reference_fragment_list()
    
    # Get the fragments for the molecule
    n_fragments, fragments = get_molecule_fragments(input_parameters['fragments'],
                                                    reference_fragment_list)
    
    # Initialize reference molecule dictionary
    reference_molecule = {'formula': input_parameters['formula']} 
    
    # Get the molecular volume
    reference_molecule['volume'] = calculate_molecular_volume(input_parameters['formula'],
                                                              input_parameters['rings'],
                                                              atomic_properties)
    
    # Calculate the molecular weight
    reference_molecule['weight'] = calculate_molecular_weight(input_parameters['formula'],
                                                              atomic_properties)

    return n_fragments, fragments, reference_molecule