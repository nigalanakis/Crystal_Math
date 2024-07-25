import ast 
import json
import numpy as np
import os 
import ccdc.search
from ccdc import io
from ccdc.crystal import PackingSimilarity
from ccdc.morphology import VisualHabit

from create_reference_fragments import create_reference_fragments
from maths import calculate_inertia
from maths import ensure_right_handed_coordinate_system
from maths import sort_eigenvectors
from structure_operations import get_lattice_vectors 
from structure_operations import get_unique_species 
from structure_operations import similarity_check

def structure_check(input_parameters,crystal,molecule):
    ''' 
    Performs a check to see if a structure is consistent with the used defined
    requirements.
    
    Parameters
    ----------
    input_parameters : dict
        A dictionary with the input parameters for the search.
    crystal : csd object
        The csd crystal for the structure.
    molecule : csd_object
        The csd molecule for the structure.
    
    Returns
    -------
    True if a structure is accepted, None otherwise.
    '''
    # Discard structures with based on the Z prime value
    if crystal.z_prime not in input_parameters['target_z_prime_values']:
        return None

    # Discard structures with unwanted space group 
    if input_parameters['target_space_groups'] != [] and crystal.spacegroup_symbol not in input_parameters['target_space_groups']:
        return None
    
    # Assign unknow bond types, add missing hydrogens and assign 
    # partial charges to atoms
    try:
        molecule.assign_bond_types()
        molecule.add_hydrogens(mode='missing')
        molecule.assign_partial_charges()
    except Exception:
        return None
    
    # Generate atoms
    try:
        atoms = molecule.atoms 
    except Exception:
        return None
    
    # Discard structures with no atoms in the crystal
    if len(atoms) == 0:
        return None
    
    # Discard structures with missing coordinates:
    for at in atoms:
        if at.coordinates == None:
            return None 
        
    # Discard structures based on the their type (homomolecular, co-crystals, hydrates)
    components = [c.formula for c in molecule.components]
    if all(item == components[0] for item in components):
        crystal_type = 'homomolecular'
    else:
        if 'H2 O1' in components:
            crystal_type = 'hydrate'
        else:
            crystal_type = 'co-crystal'    
    if crystal_type not in input_parameters['crystal_type']:
        return None
    
    # Discard structures based on formal charge of molecules
    for component in molecule.components:
        if crystal_type == 'homomolecular' and component.formal_charge not in input_parameters['molecule_formal_charges']:
            return None
    
    # Discard structures with out-of-range molecular weight
    for component in molecule.components:
        if component.molecular_weight > input_parameters['molecule_weight_limit']:
            return None

    # Discard structures with unwanted atomic species
    if input_parameters['target_species'] != []:
        for s in get_unique_species(crystal.formula):
            if s not in input_parameters['target_species']:
                return None
                
    return True

def get_refcode_families(input_parameters):
    '''
    Reads the CSD database and returns the refcode families and the structures
    for each family.
    
    Parameters
    ----------
    input_parameters : dict
        A dictionary with the user defined input parameters.
    
    Returns
    -------
    refcode_families : dict
        A dictionaty with the refcode families and the structures for each
        family.
    '''
    # Initialize the reader for the CSD
    reader = io.EntryReader('CSD')
   
    # List to hold the matching Refcodes
    refcode_families = {}
   
    # Iterate through all entries in the CSD
    family_i = ''
    for entry in reader: 
        family_j = entry.identifier[:6]
        if family_j != family_i:
            refcode_families[family_j] = [entry.identifier]
        else:
            refcode_families[family_j].append(entry.identifier)
        family_i = family_j
        
    # Specify the filename you want to write to
    filename = '../csd_db_analysis/db_data/' + input_parameters['data_prefix'] + '_csd_refcode_families.json'
    
    # Writing the dictionary to a file in JSON format
    with open(filename, 'w') as f:
        json.dump(refcode_families, f, indent=4)  

    return refcode_families

def cluster_refcode_families(input_parameters):  
    '''
    Reads the csd families distionary and returns a new dictionary for the 
    refcode familes where the structures are grouped based on their similarity.
    Only strutures consisntent with the user defined criteria are included in 
    the clustered refcode families.
    
    Parameters
    ----------
    input_parameters : dict
        A dictionary with the input parameters for the search.

    Returns
    -------
    refcode_families : dict
        A dictionaty with the refcode families and the structures for each
        family grouped based on similarity.
    '''   
    # Open the refcode families file and read data.
    refcode_families_f = '../csd_db_analysis/db_data/' + input_parameters['data_prefix'] + '_csd_refcode_families.json'
    if not os.path.exists(refcode_families_f):
        # If the file does not exist, raise an exception
        raise FileNotFoundError(f'The file {refcode_families_f} does not exist.')
    else:
        # Set the checking similarity engine
        similarity_engine = PackingSimilarity()
        similarity_engine.settings.distance_tolerance = 0.2
        similarity_engine.settings.angle_tolerance = 20.
        similarity_engine.settings.ignore_bond_types = True
        similarity_engine.settings.ignore_hydrogen_counts = True
        similarity_engine.settings.ignore_hydrogen_positions = True
        similarity_engine.settings.packing_shell_size = 15
        
        # Get the families and member structures.
        with open(refcode_families_f) as f:
            data = f.read()
        refcode_families = ast.literal_eval(data)
        
        # Get families with more than one structure. For families with one
        # one structure, add structure to the unique structure lise.
        csd_entries = io.EntryReader('CSD')
        families_clustered = {}
        for family in refcode_families:
            if len(refcode_families[family]) > 1:
                structures_to_check = []
                for structure in refcode_families[family]:
                    entry = csd_entries.entry(structure)
                    crystal = entry.crystal
                    molecule = entry.molecule 
    
                    # Check if structure is valid according to search criteria.
                    if structure_check(input_parameters,crystal,molecule) == None:
                        continue
                        
                    structures_to_check.append([structure,crystal])
    
                # Get similar structures
                similar_structure_groups = similarity_check(structures_to_check,similarity_engine)
                
                # Print out groups of similar structures
                if len(similar_structure_groups) > 0:
                    families_clustered[family] = []
                    for i, group in enumerate(similar_structure_groups):
                        group = sorted(group)
                        families_clustered[family].append(group)
                
            else:
                entry = csd_entries.entry(refcode_families[family][0])
                crystal = entry.crystal
                molecule = entry.molecule 
        
                # Check if structure is valid according to search criteria.
                if structure_check(input_parameters,crystal,molecule) == None:
                    continue
                    
                families_clustered[family] = [[refcode_families[family][0]]]
                
    # Specify the filename for the clustered families
    filename = '../csd_db_analysis/db_data/' + input_parameters['data_prefix'] + '_csd_refcode_families_clustered.json'
    
    # Writing the dictionary to a file in JSON format
    with open(filename, 'w') as f:
        json.dump(families_clustered, f, indent=4)  
    
    return families_clustered

def get_unique_structures(input_parameters):
    '''
    Goes through the clustered refcode families and return a single structure
    for each group of similar structures. The resulting structure is based on 
    user defined criteria.
    
    Parameters
    ----------
    input_parameters : dict
        A dictionary with the input parameters for the search.

    Returns
    -------
    unique_structures : dict
        A dictionaty with the unique polymorphs for each refcode family.
    '''
    # Set the unique structures clustering method
    unique_structures_clustering_method = input_parameters['unique_structures_clustering_method']
    if unique_structures_clustering_method == 'energy':
        visualhabit_settings = VisualHabit.Settings()
    
    # Open the refcode families clusters file and read data.
    refcode_families_clusters_f  = '../csd_db_analysis/db_data/' + input_parameters['data_prefix'] + '_csd_refcode_families_clustered.json'
    if not os.path.exists(refcode_families_clusters_f ):
        # If the file does not exist, raise an exception
        raise FileNotFoundError(f'The file {refcode_families_clusters_f} does not exist.')
    else:
        # Get the families and member structures.
        with open(refcode_families_clusters_f) as f:
            data = f.read()
        families_clustered = ast.literal_eval(data)
        
        # Loop over refcode family clusters.
        csd_entries = io.EntryReader('CSD')
        unique_structures = {}
        for family in families_clustered:
            # Loop over the numbe rof polymorphs 
            unique_structures[family] = []
            n_polymorphs = len(families_clustered[family])
            for i in range(n_polymorphs): 
                n_similar_structures = len(families_clustered[family][i])
                
                # If the polymorph has only one structure deposited, add 
                # structure to the dictionary. Else, cluster similar structures.
                if n_similar_structures == 1:
                    if families_clustered[family][i][0] in input_parameters['structures_to_exclude']:
                        continue
                    unique_structures[family].append(families_clustered[family][i][0])
                else:
                    # Set the minimum value for the ranking
                    minimum_value = np.inf
                    minimum_value_structure = ''
                    for structure in families_clustered[family][i]:
                        if structure in input_parameters['structures_to_exclude']:
                            continue 
                        
                        entry = csd_entries.entry(structure)
                        crystal = entry.crystal
                        
                        if unique_structures_clustering_method == 'energy':
                            try:
                                results = VisualHabit(settings=visualhabit_settings).calculate(crystal)
                            except Exception:
                                continue 
                            lattice_energy = results.lattice_energy.total
                                                        
                            if lattice_energy < minimum_value:
                                minimum_value = lattice_energy
                                minimum_value_structure = structure
                        
                        if unique_structures_clustering_method == 'vdWFV':
                            vdWFV = 1.0 - crystal.packing_coefficient
                            
                            if vdWFV < minimum_value:
                                minimum_value = vdWFV
                                minimum_value_structure = structure
                                
                    if minimum_value_structure != '': 
                        unique_structures[family].append(minimum_value_structure)
            unique_structures[family] = sorted(unique_structures[family])
            
    # Specify the filename for the clustered families
    filename = '../csd_db_analysis/db_data/' + input_parameters['data_prefix'] + '_csd_refcode_families_unique_structures.json'
    
    # Writing the dictionary to a file in JSON format
    with open(filename, 'w') as f:
        json.dump(unique_structures, f, indent=4)  

    return unique_structures

def check_for_target_fragments(input_parameters,molecule):
    fragment_list = create_reference_fragments()

    # Check for target fragments
    for fragment in fragment_list:
        if fragment not in input_parameters['target_fragments']:
            continue

        csd_fragment = ccdc.search.SMARTSSubstructure(fragment_list[fragment]['smarts'])
        fragmentSearch = ccdc.search.SubstructureSearch()
        fragmentID = fragmentSearch.add_substructure(csd_fragment)
        hits = fragmentSearch.search(molecule)

        if hits == []:
            return None
    
    return True
        
def get_csd_atom_and_molecule_properties(crystal,molecule,atoms):
    ''' 
    Extracts and returns the atomic and  molecular properties for a CSD entry. 
    
    Parameters
    ----------
    crystal : csd obj
        The CSD crystal object of the structure.
    molecule : csd obj
        The CSD molecule object of the structure.
    atoms : csd obj
        The CSD atoms object of the structure.
    
    Returns
    -------
    atom_properties : dict
        A dictionary with the atomic properties.
    molecule_properties : dict
        A dictionary with the molecular properties.
    '''
    structure_molecule = {}
    structure_molecule['atoms_charge'] = np.array([at.partial_charge for at in atoms])
    structure_molecule['atoms_labels'] = [at.label for at in atoms]
    structure_molecule['atoms_mass'] = np.round(np.array([at.atomic_weight for at in atoms]),4)
    structure_molecule['atoms_species'] = [at.atomic_symbol for at in atoms]
    structure_molecule['atoms_vdW_radius'] = np.round(np.array([at.vdw_radius for at in atoms]),4)
    structure_molecule['atoms_coordinates_f'] = np.round(np.array([[at.fractional_coordinates[i] for i in [0,1,2]] for at in atoms]),4)
    structure_molecule['atoms_coordinates_c'] = np.round(np.array([[at.coordinates[i] for i in [0,1,2]] for at in atoms]),4)
    structure_molecule['n_atoms'] = len(atoms)
    structure_molecule['coordinates_f'] = np.round(np.sum(structure_molecule['atoms_mass'].reshape(structure_molecule['n_atoms'],1) * structure_molecule['atoms_coordinates_f'],axis=0) / np.sum(structure_molecule['atoms_mass']),4)
    structure_molecule['coordinates_c'] = np.round(np.sum(structure_molecule['atoms_mass'].reshape(structure_molecule['n_atoms'],1) * structure_molecule['atoms_coordinates_c'],axis=0) / np.sum(structure_molecule['atoms_mass']),4)
    structure_molecule['volume'] = np.round(molecule.molecular_volume,4)
    structure_molecule['atoms_bond_vectors_f'] = np.round(structure_molecule['atoms_coordinates_f'] - structure_molecule['coordinates_f'],4)
    structure_molecule['atoms_bond_vectors_c'] = np.round(structure_molecule['atoms_coordinates_c'] - structure_molecule['coordinates_c'],4)
    structure_molecule['bonds'] = [[bond.atoms[0].label, bond.atoms[1].label] for bond in molecule.bonds]
    
    return structure_molecule

def get_csd_crystal_properties(crystal):
    ''' 
    Extracts and returns the crystal properties for a CSD entry. 
    
    Parameters
    ----------
    crystal : csd obj
        The CSD crystal object of the structure.
        
    Returns
    -------
    crystal_properties : dict 
        A dictionary with the crystal properties.
    '''
    # Set the engine for energy calculation 
    visualhabit_settings = VisualHabit.Settings()
    visualhabit_settings.potential = 'gavezzotti'
    try:
        energy = VisualHabit(settings=visualhabit_settings).calculate(crystal)
    except Exception:
        energy = None
    if energy != None:
        lattice_energy = energy.lattice_energy
    
    crystal_properties = {}
    crystal_properties['ID'] = crystal.identifier
    crystal_properties['formula'] = crystal.formula
    crystal_properties['species'] = get_unique_species(crystal.formula)
    crystal_properties['space_group'] = crystal.spacegroup_symbol
    crystal_properties['z_crystal'] = crystal.z_value
    crystal_properties['z_prime'] = crystal.z_prime 
    crystal_properties['cell_lengths'] = np.round(np.array([l for l in crystal.cell_lengths]),4)
    crystal_properties['scaled_cell_lengths'] = np.round(np.array([l for l in crystal.cell_lengths])/crystal.cell_lengths[0],4)
    crystal_properties['cell_angles'] = np.round(np.array([l for l in crystal.cell_angles]),2)
    crystal_properties['cell_volume'] = np.round(crystal.cell_volume,4) 
    crystal_properties['cell_density'] = np.round(crystal.calculated_density,4)
    crystal_properties['vdWFV'] = np.round(1.0 - crystal.packing_coefficient,4)
    crystal_properties['SAS'] = np.round(crystal.void_volume(probe_radius=1.2,grid_spacing=0.2,mode='accessible'),4)
    crystal_properties['lattice_vectors'] = np.round(get_lattice_vectors(crystal_properties['cell_lengths'],crystal_properties['cell_angles'],crystal_properties['cell_volume']),4)
    crystal_properties['inverse_lattice_vectors'] = np.round(get_lattice_vectors(crystal_properties['cell_lengths'],crystal_properties['cell_angles'],crystal_properties['cell_volume'],inverse=True),4)
    crystal_properties['close_contacts'] = crystal.contacts(intermolecular='Intermolecular',distance_range=(-3.0, 0.50)) 
    crystal_properties['hbonds'] = crystal.hbonds(intermolecular='Intermolecular')
    if energy != None:
        crystal_properties['lattice_energy'] = {
            'total': np.round(lattice_energy.total,4), 
            'electrostatic': np.round(lattice_energy.electrostatic,4),
            'vdW': np.round(lattice_energy.vdw,4),
            'vdW_attraction': np.round(lattice_energy.vdw_attraction,4),
            'vdW_repulsion': np.round(lattice_energy.vdw_repulsion,4),
            'h-bond': np.round(lattice_energy.h_bond,4),
            'h-bond_attraction': np.round(lattice_energy.h_bond_attraction,4),
            'h-bond_repulsion': np.round(lattice_energy.h_bond_repulsion,4)
            }
    else:
        crystal_properties['lattice_energy'] = {
            'total': 0.0, 
            'electrostatic': 0.0,
            'vdW': 0.0,
            'vdW_attraction': 0.0,
            'vdW_repulsion': 0.0,
            'h-bond': 0.0,
            'h-bond_attraction': 0.0,
            'h-bond_repulsion': 0.0}
    return crystal_properties

def get_csd_structure_fragments(input_parameters,structure,molecule):
    ''' 
    Identify and returns the fragments in a molecule 
    
    Parameters
    ----------
    input_parameters : dict
        A dictionary with the user defined input parameters.
    structure : dict
        A disctionary with the data for the structure.
    molecule : object
        The csd molecule object for the structure.
        
    Returns
    -------
    str_fragments : dict
        A dictionary with the identified fragments in the molecule
    '''
    # Update the reference fragment list 
    fragment_list = create_reference_fragments()
    
    # Get the fragments for the structure
    fragments = {}
    i_hit = 0
    for fragment in fragment_list:
        csd_fragment = ccdc.search.SMARTSSubstructure(fragment_list[fragment]['smarts'])
        fragmentSearch = ccdc.search.SubstructureSearch()
        fragmentID = fragmentSearch.add_substructure(csd_fragment)
        hits = fragmentSearch.search(molecule)
        for hit in hits:
            i_hit += 1
            key = 'F' + str(i_hit).zfill(2) + '.' + fragment
            hit_atoms = []
            hit_atoms_species = []
            hit_atoms_labels = []
            for at in hit.match_atoms():
                hit_atoms.append(structure['molecule']['atoms_labels'].index(at.label))
                hit_atoms_species.append(at.atomic_symbol)
                hit_atoms_labels.append(at.label)
            fragments[key] = {}
            fragments[key]['smarts'] = fragment_list[fragment]['smarts']
            fragments[key]['atoms'] = hit_atoms
            fragments[key]['atoms_species'] = hit_atoms_species
            fragments[key]['atoms_labels'] = hit_atoms_labels
            fragments[key]['atoms_mass'] = np.round(np.array(fragment_list[fragment]['mass']),4)
            fragments[key]['n_atoms'] = len(fragments[key]['atoms'])
            fragments[key]['atoms_coordinates_c'] = np.round(np.array(structure['molecule']['atoms_coordinates_c'][hit_atoms]),4)
            fragments[key]['atoms_coordinates_f'] = np.round(np.array(structure['molecule']['atoms_coordinates_f'][hit_atoms]),4)
            fragments[key]['atoms_coordinates_sf'] = np.round(np.array(fragment_list[fragment]['coordinates_sf']),4)
            fragments[key]['atoms_to_align'] = fragment_list[fragment]['atoms_to_align']
            fragments[key]['coordinates_c'] = np.round(np.sum(fragments[key]['atoms_mass'].reshape(fragments[key]['n_atoms'],1) * fragments[key]['atoms_coordinates_c'],axis=0) / np.sum(fragments[key]['atoms_mass']),4)
            fragments[key]['coordinates_f'] = np.round(np.sum(fragments[key]['atoms_mass'].reshape(fragments[key]['n_atoms'],1) * fragments[key]['atoms_coordinates_f'],axis=0) / np.sum(fragments[key]['atoms_mass']),4)
            fragments[key]['atoms_bond_vectors_c'] = np.round(fragments[key]['atoms_coordinates_c'] - fragments[key]['coordinates_c'],4)
            fragments[key]['atoms_bond_vectors_f'] = np.round(fragments[key]['atoms_coordinates_f'] - fragments[key]['coordinates_f'],4)
            
    # Remove subsets (sub-fragments)
    entries_to_remove = set()

    # Compare all pairs of keys
    for key1 in fragments:
        for key2 in fragments:
            if key1 != key2 and key1 not in entries_to_remove and key2 not in entries_to_remove:
                if fragments[key1]['smarts'] == fragments[key2]['smarts']:
                    continue
                
                atoms1 = set(fragments[key1]['atoms_labels'])
                atoms2 = set(fragments[key2]['atoms_labels'])
                
                # Check if atoms of entry1 are subset of entry2
                if atoms1.issubset(atoms2):
                    entries_to_remove.add(key1)
                elif atoms2.issubset(atoms1):
                    entries_to_remove.add(key2)
    
    # Remove identified keys from the dictionary
    for key in entries_to_remove:
        del fragments[key]
        
    # Add a fragment number ID
    str_fragments = {}
    for i, key in enumerate(fragments):
        new_key = 'F' + str(i + 1).zfill(2) + '.' + key[4:]
        str_fragments[new_key] = fragments[key]
        
    # Add fragments for full components 
    if input_parameters['add_full_component']:
        for i, component in enumerate(molecule.components):
            key = 'FMC.component_' + str(i + 1)
            str_fragments[key] = {}
            str_fragments[key]['atoms_labels'] =  [at.label for at in component.atoms]
            str_fragments[key]['atoms'] = [structure['molecule']['atoms_labels'].index(at.label) for at in component.atoms]
            str_fragments[key]['atoms_species'] = [at.atomic_symbol for at in component.atoms]
            str_fragments[key]['atoms_mass'] = np.round(np.array([at.atomic_weight for at in component.atoms]),4)
            str_fragments[key]['n_atoms'] = len(component.atoms)
            str_fragments[key]['atoms_coordinates_c'] = np.round(np.array([at.coordinates for at in component.atoms]),4)
            str_fragments[key]['atoms_coordinates_f'] = np.round(np.array([at.fractional_coordinates for at in component.atoms]),4)
            str_fragments[key]['atoms_to_align'] = 'all'
            str_fragments[key]['coordinates_c'] = np.round(np.sum(str_fragments[key]['atoms_mass'].reshape(str_fragments[key]['n_atoms'],1) * str_fragments[key]['atoms_coordinates_c'],axis=0) / np.sum(str_fragments[key]['atoms_mass']),4)
            str_fragments[key]['coordinates_f'] = np.round(np.sum(str_fragments[key]['atoms_mass'].reshape(str_fragments[key]['n_atoms'],1) * str_fragments[key]['atoms_coordinates_f'],axis=0) / np.sum(str_fragments[key]['atoms_mass']),4)
            str_fragments[key]['atoms_bond_vectors_c'] = np.round(str_fragments[key]['atoms_coordinates_c'] - str_fragments[key]['coordinates_c'],4)
            str_fragments[key]['atoms_bond_vectors_f'] = np.round(str_fragments[key]['atoms_coordinates_c'] - str_fragments[key]['coordinates_c'],4)
            
            # Set the rotation of the full component
            inertia_eigenvalues, inertia_eigenvectors = calculate_inertia(str_fragments[key]['atoms_mass'],str_fragments[key]['atoms_bond_vectors_c'])                
            inertia_eigenvalues, inertia_eigenvectors = sort_eigenvectors(inertia_eigenvalues,inertia_eigenvectors)
            inertia_eigenvectors = ensure_right_handed_coordinate_system(inertia_eigenvectors)
            
            str_fragments[key]['atoms_coordinates_sf']  = np.round(np.round(np.matmul(str_fragments[key]['atoms_bond_vectors_c'],inertia_eigenvectors), decimals=4),4)
            
    return str_fragments
