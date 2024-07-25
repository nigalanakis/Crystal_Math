import ast 
import itertools
import json
import numpy as np
import os 
import re
from ccdc import io

import io_operations
from csd_operations import check_for_target_fragments
from csd_operations import get_csd_atom_and_molecule_properties
from csd_operations import get_csd_crystal_properties
from csd_operations import get_csd_structure_fragments
from get_structures_list import get_structures_list 
from maths import align_structures
from maths import cartesian_to_spherical
from maths import distance_to_plane
from maths import distance_to_zzp_planes_family
from maths import get_reference_cell_points
from maths import kabsch_rotation_matrix
from maths import set_zzp_planes
from maths import vectors_closest_to_perpendicular

class NumpyArrayEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def format_lists(json_str):
    """ Formats lists in the JSON string to remove unnecessary whitespace and newlines. """
    pattern = re.compile(r'\[\s*((?:[^[\]]|\n)+?)\s*\]', re.DOTALL)
    return re.sub(pattern, lambda x: '[' + x.group(1).replace('\n', '').replace(' ', '') + ']', json_str)

def convert_to_json(data):
    """ Converts Python dictionary to formatted JSON string. """
    json_str = json.dumps(data, cls=NumpyArrayEncoder, indent=4)
    formatted_json = format_lists(json_str)
    return formatted_json

def get_structure_data(input_parameters):
    ''' 
    Extracts data from the unique CSD structures.
    
    Parameters
    ----------
    input_parameters : dict
        A dictionary with the input parameters for the search.

    Returns
    -------
    '''
    # Set the files to write data
    db_folder = "../csd_db_analysis/db_data/"
    prefix = input_parameters["data_prefix"] 
    db_structures_folder = db_folder + "_".join([prefix,"structures"]) + "/"
    
    # Create the structures folder
    os.makedirs(db_structures_folder, exist_ok=True)
    
    # Get the reference structures dictionary.
    if input_parameters["structure_list"][0] == "csd-all":
        reference_structures_f  = '../csd_db_analysis/db_data/' + input_parameters["data_prefix"] + '_csd_refcode_families_clustered.json'
    elif input_parameters["structure_list"][0] == "csd-unique":
        reference_structures_f  = '../csd_db_analysis/db_data/' + input_parameters["data_prefix"] + '_csd_refcode_families_unique_structures.json'
    elif input_parameters["structure_list"][0] == "cif":
        cif_files_f = '../source_data/cif_files/'
        reference_structures_f  = cif_files_f + 'cif_structures_list.json'
    
    # Check if the dictionary exists.
    if not os.path.exists(reference_structures_f):
        # If the file does not exist, raise an exception
        raise FileNotFoundError(f"The file {reference_structures_f} does not exist.")
    else:
        # Get the families and member structures.
        with open(reference_structures_f) as f:
            data = f.read()
        reference_structures = ast.literal_eval(data)
           
    # Get the structures list for the analysis
    structures_list = get_structures_list(input_parameters,reference_structures)
    
    # Get the csd entries if necessary
    if input_parameters["structure_list"][0] in ["csd-all","csd-unique"]:
        csd_entries = io.EntryReader("CSD")
    
    # Initialize cell reference points 
    cell_reference_points = get_reference_cell_points(-1, 2.5, 0.5)
    
    # Set the ZZP planes
    zzp_planes = set_zzp_planes()

    # Loop over the structures in the list
    for structure_name in structures_list:
        # Set the csd_crystal and csd_molecule objects
        if input_parameters["structure_list"][0] in ["csd-all","csd-unique"]:
            entry = csd_entries.entry(structure_name)
            crystal = entry.crystal 
            if input_parameters["center_molecule"]:
                crystal.centre_molecule() # Move molecule inside unit cell 
            molecule = entry.molecule
        elif input_parameters["structure_list"][0] == "cif":
            crystal = io.CrystalReader(cif_files_f + structure_name)
            crystal = crystal[0]
            if input_parameters["center_molecule"]:
                crystal.centre_molecule() # Move molecule inside unit cell 
            molecule = io.MoleculeReader(cif_files_f + structure_name)
            molecule = molecule[0]  
            
        # Exclude structures
        if crystal.identifier in input_parameters["structures_to_exclude"]:
            continue
        
        # Add missing hydrogen atoms
        try:
            molecule.assign_bond_types()
            molecule.add_hydrogens(mode='missing')
            molecule.assign_partial_charges()
        except Exception:
            continue 
        
        # Set the atoms for the reference molecule 
        try:
            atoms = molecule.atoms
        except Exception:
            continue 
        
        # Check for unnatural atoms with no coordinates
        discard = False
        for at in atoms:
            if at.coordinates == None:
                discard = True 
                break 
            
        if discard:
            continue
        
        # Check for target fragments
        if check_for_target_fragments(input_parameters,molecule) == None:
            continue
        
        # Initialize structure
        structure = {} 
        
        # Get crystal, molecule and atom properties
        structure["crystal"] = get_csd_crystal_properties(crystal)
        structure["molecule"] = get_csd_atom_and_molecule_properties(crystal,molecule,atoms)
            
        # Get the fragments for the structure
        structure["fragments"] = get_csd_structure_fragments(input_parameters,structure,molecule)
    
        # Discard structures with none of the desired substructures 
        if not bool(structure["fragments"]):
            continue
           
        # Calculate the structure specific cell reference points in cartesian
        # coordinates
        cell_points = np.dot(cell_reference_points, structure["crystal"]["lattice_vectors"])
        
        # Loop over all fragments to calculate the fragment orientation
        # print('Analyzing structure ' + structure["crystal"]["ID"])
        for fragment in structure["fragments"]:
            current_fragment = structure["fragments"][fragment]
            # Get the list of atoms that are used for the aligmnent
            if current_fragment["atoms_to_align"] == "all":
                atoms_to_align = list(range(current_fragment["n_atoms"]))
            else:
                atoms_to_align = current_fragment["atoms_to_align"]
            
            # Get the rotation matrix
            current_fragment["rotation_matrix"] = np.round(kabsch_rotation_matrix(current_fragment["atoms_coordinates_sf"][atoms_to_align],
                                                                                  current_fragment["atoms_bond_vectors_c"][atoms_to_align]),4)
            current_fragment["inverse_rotation_matrix"] = np.round(current_fragment["rotation_matrix"].T,4)
            
            # Filter unwanted fragments in case of identical smarts representation
            if fragment[4:] in input_parameters["fragments_to_check_alignment"]:
                rmsd = align_structures(current_fragment["rotation_matrix"], 
                                        current_fragment["atoms_coordinates_sf"][atoms_to_align], 
                                        current_fragment["atoms_bond_vectors_c"][atoms_to_align])

                if rmsd > input_parameters["alignment_tolerance"]:
                    continue 
            
            # Calculate the normalized vectors perpendicular to the 
            # principal inertia planes in the crystallographic coordinates 
            # system
            current_fragment["principal_inertia_planes_f"] = np.dot(current_fragment["rotation_matrix"], structure["crystal"]["lattice_vectors"].T)
            current_fragment["principal_inertia_planes_f"] = np.round(current_fragment["principal_inertia_planes_f"] / np.linalg.norm(current_fragment["principal_inertia_planes_f"], axis=1, keepdims=True),4)
           
            # Identify for each eigenvector the proposed vectors that are
            # closest to be perpendicular and the respective angle
            current_fragment["n_max_vectors"] = vectors_closest_to_perpendicular(current_fragment["principal_inertia_planes_f"], 
                                                                                 input_parameters["proposed_vectors_n_max"])
        
            # Calculate minimum distances of pripcipal inertia planes to the 
            # corners of all the points of a 3x3x3 supercell in the form
            # (0.5k1, 0.5k2, 0.5k3), k1, k2, k3 = -2, -1, ..., 4
            minimum_distances_to_planes = []
            for plane in current_fragment["rotation_matrix"]:
                d_min = np.inf
                for point in cell_points:
                    d = distance_to_plane(point,plane,current_fragment["coordinates_c"],normal=False)
                    if d < d_min:
                        d_min = d
                minimum_distances_to_planes.append(d_min)
            current_fragment["principal_inertia_planes_distances_to_cell_points"] = np.round(minimum_distances_to_planes,4)
            
            # Calculate minimum distance of non-hydeogen atoms to ZZP planes
            minimum_distances_to_zzp_planes = []
            for point in current_fragment["atoms_coordinates_f"]:
                d_min = np.inf 
                for plane_normal, plane_norm in zzp_planes:
                    d = distance_to_zzp_planes_family(point, plane_normal, plane_norm)
                    if d < d_min:
                        d_min = d
                minimum_distances_to_zzp_planes.append(d_min)
            current_fragment["minimum_atom_distances_to_zzp_planes"] = np.round(minimum_distances_to_zzp_planes,4)
            
            # Add hydrogen atoms to fragmetnts
            if fragment[:3] != "FMC":
                for atom, atom_label in zip(structure["molecule"]["atoms_species"],structure["molecule"]["atoms_labels"]):
                    if atom == "H":
                        for at1, at2 in structure["molecule"]["bonds"]:
                            if at1 == atom_label:
                                bonded_atom = at2
                            if at2 == atom_label:
                                bonded_atom = at1
                        if bonded_atom in current_fragment["atoms_labels"]:
                            current_fragment["atoms_species"].append(atom)
                            current_fragment["atoms_labels"].append(atom_label)
        
        # Create the contacts dictionary
        structure_contacts = {}
        for contact in structure["crystal"]["close_contacts"]:
            # Check if the contact is part of an h-bond
            is_hbond = False
            for hbond in structure["crystal"]["hbonds"]:
                hbond_atom_labels = [atom.label for atom in hbond.atoms]
                if (contact.atoms[0].label, contact.atoms[1].label) in list(itertools.permutations(hbond_atom_labels,2)):    
                    if [contact.atoms[0].label, contact.atoms[1].label] not in structure["molecule"]["bonds"] and [contact.atoms[1].label, contact.atoms[0].label] not in structure["molecule"]["bonds"]:                    
                        is_hbond = True 
                        break

            # Get the central and contact groups (fragments)
            central_group = [fragment for fragment in structure["fragments"] if contact.atoms[0].label in structure["fragments"][fragment]["atoms_labels"] if fragment[:3] != "FMC"]
            contact_group = [fragment for fragment in structure["fragments"] if contact.atoms[1].label in structure["fragments"][fragment]["atoms_labels"] if fragment[:3] != "FMC"]
            for i in [0, 1]: 
                for fragment1 in central_group:
                    for fragment2 in contact_group:
                        at1, at2 = 0, 1
                        central_fragment, contact_fragment = fragment1, fragment2 
                            
                        # Get the bond vectors of the contact atoms to the central 
                        # fragment
                        central_bond_vector = contact.atoms[at1].coordinates - structure["fragments"][central_fragment]["coordinates_c"]
                        contact_bond_vector = contact.atoms[at2].coordinates - structure["fragments"][central_fragment]["coordinates_c"]
                        
                        # Rotate them to the central fragment's reference system
                        central_bond_vector_r = np.dot(central_bond_vector,structure["fragments"][central_fragment]["inverse_rotation_matrix"])
                        contact_bond_vector_r = np.dot(contact_bond_vector,structure["fragments"][central_fragment]["inverse_rotation_matrix"])
                        
                        # Convert contact bond vector to spherical coodinates
                        contact_bond_vector_spherical = cartesian_to_spherical(contact_bond_vector_r)
                        
                        # Get the contact type
                        contact_type = "hbond" if is_hbond else "vdW"
                       
                        # Add contact data to list
                        structure_contacts['_'.join([contact.atoms[at1].label,fragment1,contact.atoms[at2].label,fragment2])] = {
                            "cc_length": np.round(contact.length,4),
                            "cc_type": contact_type,
                            "cc_is_in_los": contact.is_in_line_of_sight,
                            "cc_central_atom": {
                                "atom": contact.atoms[at1].atomic_symbol,
                                "fragment": central_fragment[4:],
                                "coordinates": {
                                    "cartesian": np.round(contact.atoms[at1].coordinates,4),
                                    "fractional": np.round(contact.atoms[at1].fractional_coordinates,4)
                                    },
                                "bond_vectors": np.round(central_bond_vector,4),
                                "reference_bond_vectors": np.round(central_bond_vector_r,4)
                                },
                            "cc_contact_atom": {
                                "atom": contact.atoms[at2].atomic_symbol,
                                "fragment": contact_fragment[4:],
                                "coordinates": {
                                    "cartesian": np.round(contact.atoms[at2].coordinates,4),
                                    "fractional": np.round(contact.atoms[at2].fractional_coordinates,4)
                                    },
                                "bond_vectors": np.round(contact_bond_vector,4),
                                "reference_bond_vectors": np.round(contact_bond_vector_r,4),
                                "reference_bond_vectors_spherical": np.round(contact_bond_vector_spherical,4)
                                },
                            }
        structure["crystal"]["close_contacts"] = structure_contacts

        # Create the hydrogen bonds dictionary
        structure_hbonds = {}
        for hbond in structure["crystal"]["hbonds"]:
            # Get the donor atom
            hbond_atom_labels = [atom.label for atom in hbond.atoms]
            for bond in structure["molecule"]["bonds"]:
                if hbond_atom_labels[1] in bond:
                    if hbond_atom_labels[0] in bond:
                        hbond_donor = 0
                        hbond_acceptor = 2
                    if hbond_atom_labels[2] in bond:
                        hbond_donor = 2
                        hbond_acceptor = 0
            
            structure_hbonds['_'.join([hbond.atoms[hbond_donor].label,hbond.atoms[1].label,hbond.atoms[hbond_acceptor].label])] = {
                "hb_atoms": (hbond.atoms[hbond_donor].atomic_symbol,hbond.atoms[1].atomic_symbol,hbond.atoms[hbond_acceptor].atomic_symbol),
                "hb_length": np.round(hbond.length,4),
                "hb_da_distance": np.round(hbond.da_distance,4),
                "hb_angle": np.round(hbond.angle,4),
                "hb_is_in_los": hbond.is_in_line_of_sight,
                "hb_donor_coordinates": np.round(hbond.atoms[hbond_donor].coordinates,4),
                "hb_h_coordinates": np.round(hbond.atoms[1].coordinates,4),
                "hb_acceptor_coordinates": np.round(hbond.atoms[hbond_acceptor].coordinates,4),
                }
        structure["crystal"]["hbonds"] = structure_hbonds
        
        # Create the crystal dictionary
        structure_crystal = {
            "str_id": structure["crystal"]["ID"],
            "space_group": structure["crystal"]["space_group"],
            "z_crystal": structure["crystal"]["z_crystal"],
            "z_prime": structure["crystal"]["z_prime"],
            "formula": structure["crystal"]["formula"],
            "species": structure["crystal"]["species"],
            "cell_lengths": structure["crystal"]["cell_lengths"],
            "scaled_cell_lengths": structure["crystal"]["scaled_cell_lengths"],
            "cell_angles": structure["crystal"]["cell_angles"],
            "cell_volume": structure["crystal"]["cell_volume"],
            "cell_density": structure["crystal"]["cell_density"],
            "vdWFV": structure["crystal"]["vdWFV"],
            "SAS": structure["crystal"]["SAS"],
            "lattice_vectors": structure["crystal"]["lattice_vectors"],
            "lattice_energy": structure["crystal"]["lattice_energy"],
            "close_contacts": structure["crystal"]["close_contacts"],
            "hbonds": structure["crystal"]["hbonds"],
            }
            
        # Create the fragments dictionary 
        structure_fragments = {}
        for fragment in structure["fragments"]:
            # Get the data for the atoms
            at_labels = structure["fragments"][fragment]["atoms_labels"]
            at_species = structure["fragments"][fragment]["atoms_species"]
            at_coordinates_c = structure["fragments"][fragment]["atoms_coordinates_c"]
            at_coordinates_f = structure["fragments"][fragment]["atoms_coordinates_f"]
            at_bond_vectors_c = structure["fragments"][fragment]["atoms_bond_vectors_c"]
            at_bond_vectors_f = structure["fragments"][fragment]["atoms_bond_vectors_f"]
            min_distance_to_zzp = structure["fragments"][fragment]["minimum_atom_distances_to_zzp_planes"]
            fragment_atoms = {}
            for label, species, coor_c, coor_f, bv_c, bv_f, d_min in zip(at_labels,at_species,at_coordinates_c,at_coordinates_f,at_bond_vectors_c,at_bond_vectors_f,min_distance_to_zzp):
                fragment_atoms[label] = {
                    "species": species,
                    "coordinates": {
                        "cartesian": coor_c,
                        "fractional": coor_f    
                        },
                    "bond_vectors": {
                        "cartesian": bv_c,
                        "fractional": bv_f    
                        },
                    "dzzp_min": d_min
                    }
                
            # Get the data for the inertia planes
            fragment_inertia_planes = {}
            eigvecs_c = structure["fragments"][fragment]["rotation_matrix"]
            eigvecs_f = structure["fragments"][fragment]["principal_inertia_planes_f"]
            n_max_vectors = structure["fragments"][fragment]["n_max_vectors"]
            eigvecs_dmin = structure["fragments"][fragment]["principal_inertia_planes_distances_to_cell_points"]
            for i_vector, (e, w, (_, ((n1, ang1), (n2, ang2))), d_min) in enumerate(zip(eigvecs_c,eigvecs_f,n_max_vectors,eigvecs_dmin)):
                fragment_inertia_planes["e_" + str(i_vector + 1)] = {
                    "cartesian": e,
                    "crystallographic": w,
                    "perpendicular_vectors": {
                        "vector_1": n1,
                        "vector_2": n2,
                        "angle_1": ang1,
                        "angle_2": ang2},
                    "min_distance_to reference_points": d_min
                    }
             
            # Set the fragment name
            if fragment[4:-2] == "component":
                fragment_name = "component"
            else:
                fragment_name = fragment[4:]
                
            # Set the fragment data
            structure_fragments[fragment] = {
                "fragment": fragment_name,
                "coordinates": {
                    "cartesian": structure["fragments"][fragment]["coordinates_c"],
                    "fractional": structure["fragments"][fragment]["coordinates_f"]
                    },
                "inertia_planes": fragment_inertia_planes,
                "atoms": fragment_atoms
                }
            
        # Set the complete structure data
        structure_data = {
            "crystal": structure_crystal,
            "fragments": structure_fragments
            }
        
        # Convert data to json format
        structure_data = convert_to_json(structure_data)
        
        # Write data to file
        io_operations.write_structure_data_file(db_structures_folder,structure_crystal,structure_data)
        
    return
    
def get_structure_filter_data(input_parameters):
    """
    Creates a dictionary with structure information that can be used to rapidly 
    filter structures for analysis

    Parameters
    ----------
    input_parameters : dict
        A dictionary with the input parameters for the search.

    Returns
    -------
    None.

    """
    # Set the files to read and write data
    db_folder = "../csd_db_analysis/db_data/"
    prefix = input_parameters["data_prefix"] 
    db_structures_folder = db_folder + "_".join([prefix,"structures"]) + "/"
    
    # Read the structures list 
    structures_list = os.listdir(db_structures_folder)
    
    # Get the structure filter data
    structures_filter_data = {}
    for structure in structures_list:
        with open(db_structures_folder + "/" + structure,"r") as f:
            structure_data = json.load(f)
                        
            structure_crystal = structure_data["crystal"]
            structure_fragments = structure_data["fragments"]
            structure_contacts = structure_crystal["close_contacts"]
            
            fragments = []
            for fragment in structure_fragments:
                if structure_fragments[fragment]["fragment"] not in fragments:
                    fragments.append(structure_fragments[fragment]["fragment"])
                    
            contact_pairs = []
            contact_central_fragments = []
            contact_fragment_pairs = []
            for contact in structure_contacts:
                contact_pair = [structure_contacts[contact]["cc_central_atom"]["atom"],structure_contacts[contact]["cc_contact_atom"]["atom"],structure_contacts[contact]["cc_type"],structure_contacts[contact]["cc_is_in_los"]]
                if contact_pair not in contact_pairs:
                    contact_pairs.append(contact_pair)
                    
                contact_central_fragment = [structure_contacts[contact]["cc_central_atom"]["fragment"],structure_contacts[contact]["cc_type"],structure_contacts[contact]["cc_is_in_los"]]
                if contact_central_fragment not in contact_central_fragments:
                    contact_central_fragments.append(contact_central_fragment)
                
                contact_fragment_pair = [structure_contacts[contact]["cc_central_atom"]["fragment"],structure_contacts[contact]["cc_contact_atom"]["fragment"],structure_contacts[contact]["cc_type"],structure_contacts[contact]["cc_is_in_los"]]
                if contact_fragment_pair not in contact_fragment_pairs:
                    contact_fragment_pairs.append(contact_fragment_pair)
            
            structures_filter_data[structure_data["crystal"]["str_id"]] = {
                "space_group": structure_crystal["space_group"],
                "z_crystal": structure_crystal["z_crystal"],
                "z_prime": structure_crystal["z_prime"],
                "species": structure_crystal["species"],
                "fragments": fragments,
                "contact_pairs": contact_pairs,
                "contact_central_fragments": contact_central_fragments,
                "contact_fragment_pairs": contact_fragment_pairs
                }
        
    # Convert data to json format
    structures_filter_data = convert_to_json(structures_filter_data)

    # Write data to file
    io_operations.write_structures_filter_data(input_parameters,structures_filter_data)
    
    return
        
                        
                        

    