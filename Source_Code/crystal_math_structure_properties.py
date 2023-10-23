import numpy as np
import re
import crystal_math_maths as maths

def crystal_properties(crystal):
    """ 
    Extracts and returns the crystal properties for a CSD entry. 
    
    Parameters: 
        crystal (obj): The CSD crystal object of the structure.
        
    Returns:
        crystal_properties (dict): A dictionary with the crystal properties.
    """
    crystal_properties = {}
    crystal_properties["ID"] = crystal.identifier
    crystal_properties["species"] = unique_species(crystal.formula)
    crystal_properties["space_group"] = crystal.spacegroup_symbol
    crystal_properties["z_crystal"] = crystal.z_value
    crystal_properties["z_prime"] = crystal.z_prime 
    crystal_properties["cell_lengths"] = np.array([l for l in crystal.cell_lengths])
    crystal_properties["scaled_cell_lengths"] = np.array([l for l in crystal.cell_lengths])/crystal.cell_lengths[0]
    crystal_properties["cell_angles"] = np.array([l for l in crystal.cell_angles])
    crystal_properties["cell_volume"] = crystal.cell_volume 
    crystal_properties["cell_density"] = crystal.calculated_density
    crystal_properties["vdWFV"] = crystal.void_volume(probe_radius=0.0,grid_spacing=0.2,mode='contact') 
    crystal_properties["SAS"] = crystal.void_volume(probe_radius=0.5,grid_spacing=0.2,mode='accessible') 
    crystal_properties["h-matrix"] = maths.crystal_h_matrix(crystal_properties["cell_lengths"],crystal_properties["cell_angles"],crystal_properties["cell_volume"])
    crystal_properties["t-matrix"] = maths.crystal_t_matrix(crystal_properties["cell_lengths"],crystal_properties["cell_angles"],crystal_properties["cell_volume"])
    crystal_properties["close_contacts"] = crystal.contacts(intermolecular='Intermolecular',distance_range=(-3.0, 0.60)) 
    crystal_properties["h-bonds"] = crystal.hbonds(intermolecular='Intermolecular')
    
    return crystal_properties

def atom_and_molecule_properties(crystal,molecule,atoms):
    """ 
    Extracts and returns the atomic and  molecular properties for a CSD entry. 
    
    Parameters:
        crystal (obj): The CSD crystal object of the structure.
        molecule (obj): The CSD molecule object of the structure.
        atoms (obj): The CSD atoms object of the structure.
    
    Returns:
        atom_properties (dict): A dictionary with the atomic properties.
        molecule_properties (dict): A dictionary with the molecular properties.
    """
    atom_properties = {}
    molecule_properties = {}
    atom_properties["charge"] = np.array([at.partial_charge for at in atoms])
    atom_properties["label"] = [at.label for at in atoms]
    atom_properties["mass"] = np.array([at.atomic_weight for at in atoms])
    atom_properties["species"] = [at.atomic_symbol for at in atoms]
    atom_properties["vdW_radius"] = np.array([at.vdw_radius for at in atoms])
    atom_properties["coordinates_f"] = np.array([[at.fractional_coordinates[i] for i in [0,1,2]] for at in atoms])
    atom_properties["coordinates_c"] = np.array([[at.coordinates[i] for i in [0,1,2]] for at in atoms])
    molecule_properties["n_atoms"] = len(atoms)
    molecule_properties["coordinates_f"] = np.sum(atom_properties["mass"].reshape(molecule_properties["n_atoms"],1) * atom_properties["coordinates_f"],axis=0) / np.sum(atom_properties["mass"])
    molecule_properties["coordinates_c"] = np.sum(atom_properties["mass"].reshape(molecule_properties["n_atoms"],1) * atom_properties["coordinates_c"],axis=0) / np.sum(atom_properties["mass"])
    molecule_properties["volume"] = molecule.molecular_volume
    atom_properties["bond_vectors_f"] = atom_properties["coordinates_f"] - molecule_properties["coordinates_f"]
    atom_properties["bond_vectors_c"] = atom_properties["coordinates_c"] - molecule_properties["coordinates_c"]

    return atom_properties, molecule_properties

def unique_species(molecule):
    """
    Extracts and returns unique species (elements) from a molecular formula string
    in alphabetical order.

    Parameters:
        molecule (str): The molecular formula as a string.

    Returns:
        list of str: Unique species in alphabetical order.
    """
    # Regex pattern to extract element symbols
    pattern = re.compile(r'([A-Z][a-z]?)(\d+)?')
    
    # Extracting elements
    elements = pattern.findall(molecule)
    
    # Extracting unique element symbols and sorting them
    unique_species = sorted(set([element[0] for element in elements]))
    
    return unique_species