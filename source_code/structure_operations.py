import networkx as nx 
import numpy as np
import re 
from collections import defaultdict

def get_unique_species(molecule):
    """
    Extracts and returns unique species (elements) from a molecular formula string
    in alphabetical order.

    Parameters
    ----------
    molecule : str
        The molecular formula as a string.

    Return
    ------
    list of str
        Unique species in alphabetical order.
    """
    # Regex pattern to extract element symbols
    pattern = re.compile(r'([A-Z][a-z]?)(\d+)?')
    
    # Extracting elements
    elements = pattern.findall(molecule)
    
    # Extracting unique element symbols and sorting them
    unique_species = sorted(set([element[0] for element in elements]))
    
    return unique_species

def get_atoms_from_formula(formula):
    '''
    Reads the atomic formula and returns the number of atoms for each species 

    Parameters
    ----------
    formula : str
        The molecular formula.

    Returns
    -------
    species_counts : dict
        A dictionary with the count for each species.
    n_atoms : int
        The total number of atoms in the molecule.
    '''
    # This regular expression will match speciess and their counts
    species_regex = r'([A-Z][a-z]?)(\d*)'

    # Use defaultdict to handle speciess with no specified count (count of 1)
    species_counts = defaultdict(int)
    n_atoms = 0
    n_heavy_atoms = 0

    # Find all matches of the species regex in the formula
    for species, count in re.findall(species_regex, formula):
        # If count is empty, it means the species count is 1
        species_count = int(count) if count else 1
        # Add the count to the species in the dictionary
        species_counts[species] += species_count
        # Add the count to the total number of atoms
        n_atoms += species_count
        # Add the count to the total number of heavy atoms
        if species != 'H':
            n_heavy_atoms += species_count

    return dict(species_counts), n_atoms, n_heavy_atoms
    
def similarity_check(structures,similarity_engine):
    '''
    Performs a similarity check between a group of structures

    Parameters
    ----------
    structures : dict
        A dictionary with the structures to check.
    similarity_engine : obj
        The csd python API similarity check engine.

    Returns
    -------
    similar_structure_groups : list
        A list with groups of similar structures.

    '''
    # Create a new graph for the structures to be checked
    G = nx.Graph()
    
    # Add nodes for each structure
    for structure, _ in structures:
        G.add_node(structure)
    
    for i1, (structure1,crystal1) in enumerate(structures):
        for i2, (structure2,crystal2) in enumerate(structures):
            if i1 >= i2:
                continue

            try:
                h = similarity_engine.compare(crystal1, crystal2)
            except RuntimeError:
                h = None
            
            if h == None:
                continue 
            
            # If structures meet similarity criteria, add an edge
            if h.nmatched_molecules == 15 and h.rmsd < 1.0:
                G.add_edge(structure1, structure2)
                
    # Find groups of similar structures
    # Each set in 'similar_groups' contains structures that are considered similar
    similar_structure_groups = list(nx.connected_components(G))
    
    return similar_structure_groups

def get_lattice_vectors(cell_lengths,cell_angles,cell_volume,inverse=False):
    ''' 
    Calculates and returns the coordinate transformation matrices .
    
    Parameters
    ----------
    cell_lengths : numpy.ndarray
        The cell lengths of the unit cell.
    cell_angles : numpy.ndarray
        The cell angles of the unit cell.
    cell_volume : float
        The volume of the unit cell.
        
    Returns
    -------
    numpy.ndarray
        The transformation matrix from Cartesian to fractional (3,3) 
    '''
    # Set the individual cell lengths and angles in radians
    a, b, c = cell_lengths
    alpha, beta, gamma = cell_angles * np.pi / 180.0
    
    # Calculate trigomometric numbers for the angles
    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)
    
    if inverse:
        return np.array([[1.0 / a, -cos_gamma / a / sin_gamma, b * c * (cos_alpha * cos_gamma - cos_beta) / cell_volume / sin_gamma],
                         [    0.0,        1.0 / b / sin_gamma, a * c * (cos_beta * cos_gamma - cos_alpha) / cell_volume / sin_gamma],
                         [    0.0,                        0.0,                                      a * b * sin_gamma / cell_volume]]).T
        
    else:
        return np.array([[   a, b * cos_gamma,                                       c * cos_beta],
                         [ 0.0, b * sin_gamma, c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma],
                         [ 0.0,           0.0,                    cell_volume / a / b / sin_gamma]]).T

