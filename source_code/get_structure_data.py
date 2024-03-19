import ast 
import itertools
import numpy as np
import os 
from ccdc import io

import io_operations
from csd_operations import get_csd_atom_and_molecule_properties
from csd_operations import get_csd_crystal_properties
from csd_operations import get_csd_structure_fragments
from csd_operations import structure_check
from get_structures_list import get_structures_list 
from maths import align_structures
from maths import cartesian_to_spherical
from maths import distance_to_plane
from maths import distance_to_zzp_planes_family
from maths import get_reference_cell_points
from maths import kabsch_rotation_matrix
from maths import set_zzp_planes
from maths import vectors_closest_to_perpendicular

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
    structure_data_file = io_operations.check_for_file(db_folder, prefix + "_structure_data.txt")
    fragments_data_file = io_operations.check_for_file(db_folder, prefix + "_fragment_data.txt")
    contacts_data_file = io_operations.check_for_file(db_folder, prefix + "_contacts_data.txt")
    hbonds_data_file = io_operations.check_for_file(db_folder, prefix + "_hbond_data.txt")
    
    # Create file headers
    io_operations.create_file_headers(structure_data_file,
                                      fragments_data_file,
                                      contacts_data_file,
                                      hbonds_data_file)
    
    # Get the reference structures dictionary.
    if input_parameters["structure_list"][0] == "csd-all":
        reference_structures_f  = '../csd_db_analysis/db_data/csd_refcode_families_clustered.json'
    elif input_parameters["structure_list"][0] == "csd-unique":
        reference_structures_f  = '../csd_db_analysis/db_data/csd_refcode_families_unique_structures.json'
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
        print('Analyzing structure ' + structure["crystal"]["ID"])
        for fragment in structure["fragments"]:
            current_fragment = structure["fragments"][fragment]
            # Get the list of atoms that are used for the aligmnent
            if current_fragment["atoms_to_align"] == "all":
                atoms_to_align = list(range(current_fragment["n_atoms"]))
            else:
                atoms_to_align = current_fragment["atoms_to_align"]
            
            # Get the rotation matrix
            current_fragment["rotation_matrix"] = kabsch_rotation_matrix(current_fragment["atoms_coordinates_sf"][atoms_to_align],
                                                                          current_fragment["atoms_bond_vectors_c"][atoms_to_align])
            current_fragment["inverse_rotation_matrix"] = current_fragment["rotation_matrix"].T
            
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
            current_fragment["principal_inertia_planes_f"] = current_fragment["principal_inertia_planes_f"] / np.linalg.norm(current_fragment["principal_inertia_planes_f"], axis=1, keepdims=True)
           
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
            current_fragment["principal_inertia_planes_distances_to_cell_points"] = minimum_distances_to_planes
            
            # Calculate minimum distance of non-hydeogen atoms to ZZP planes
            minimum_distances_to_zzp_planes = []
            for point in current_fragment["atoms_coordinates_f"]:
                d_min = np.inf 
                for plane_normal, plane_norm in zzp_planes:
                    d = distance_to_zzp_planes_family(point, plane_normal, plane_norm)
                    if d < d_min:
                        d_min = d
                minimum_distances_to_zzp_planes.append(d_min)
            current_fragment["minimum_atom_distances_to_zzp_planes"] = minimum_distances_to_zzp_planes 
            
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
            
        # Get detailed contact data
        structure_contacts = []
        for contact in structure["crystal"]["close_contacts"]:
            # Check if the contact is part of an h-bond
            is_hbond = False
            for hbond in structure["crystal"]["h-bonds"]:
                hbond_atom_labels = [atom.label for atom in hbond.atoms]
                if (contact.atoms[0].label, contact.atoms[1].label) in list(itertools.permutations(hbond_atom_labels,2)):    
                    if [contact.atoms[0].label, contact.atoms[1].label] not in structure["molecule"]["bonds"]:                    
                        is_hbond = True 
                        break
                
            # Get the central and contact groups (fragments)
            central_group = [fragment for fragment in structure["fragments"] if contact.atoms[0].label in structure["fragments"][fragment]["atoms_labels"] if fragment[:3] != "FMC"]
            contact_group = [fragment for fragment in structure["fragments"] if contact.atoms[1].label in structure["fragments"][fragment]["atoms_labels"] if fragment[:3] != "FMC"]
            for i in [0,1]: 
                for fragment1 in central_group:
                    for fragment2 in contact_group:
                        if i == 0:
                            at1, at2 = 0, 1
                            central_fragment, contact_fragment = fragment1, fragment2 
                        else: 
                            at1, at2 = 1, 0
                            central_fragment, contact_fragment = fragment2, fragment1
                            
                        # Get the bond vectors of the contact atoms to the central 
                        # fragment
                        central_bond_vector = contact.atoms[at1].coordinates - structure["fragments"][central_fragment]["coordinates_c"]
                        contact_bond_vector = contact.atoms[at2].coordinates - structure["fragments"][central_fragment]["coordinates_c"]
                        
                        # Rotate them to the central fragment's reference system
                        central_bond_vector_r = np.dot(central_bond_vector,structure["fragments"][central_fragment]["inverse_rotation_matrix"])
                        contact_bond_vector_r = np.dot(contact_bond_vector,structure["fragments"][central_fragment]["inverse_rotation_matrix"])
                        
                        # Convert contact bond vector to spherical coodinates
                        contact_bond_vector_spherical = cartesian_to_spherical(contact_bond_vector_r)
                        
                        # Add contact data to list
                        structure_contacts.append([contact.atoms[at1].label,
                                                    contact.atoms[at2].label,
                                                    contact.atoms[at1].atomic_symbol,
                                                    contact.atoms[at2].atomic_symbol,
                                                    is_hbond,
                                                    central_fragment[4:],
                                                    contact_fragment[4:],
                                                    np.round(contact.length,4),
                                                    contact.is_in_line_of_sight,
                                                    *np.round(contact.atoms[at1].coordinates,4),
                                                    *np.round(contact.atoms[at2].coordinates,4),
                                                    *np.round(central_bond_vector,4),
                                                    *np.round(contact_bond_vector,4),
                                                    *np.round(central_bond_vector_r,4),
                                                    *np.round(contact_bond_vector_r,4),
                                                    *np.round(contact_bond_vector_spherical,4)])
        structure["crystal"]["close_contacts"] = structure_contacts
        
        # Get detailed h-bond data
        structure_hbonds = []
        for hbond in structure["crystal"]["h-bonds"]:
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
                        
            structure_hbonds.append([hbond.atoms[hbond_donor].label,
                                     hbond.atoms[1].label,
                                     hbond.atoms[hbond_acceptor].label,
                                     hbond.atoms[hbond_donor].atomic_symbol,
                                     hbond.atoms[1].atomic_symbol,
                                     hbond.atoms[hbond_acceptor].atomic_symbol,
                                     np.round(hbond.length,4),
                                     np.round(hbond.da_distance,4),
                                     np.round(hbond.angle,4),
                                     hbond.is_in_line_of_sight,
                                     *np.round(hbond.atoms[hbond_donor].coordinates,4),
                                     *np.round(hbond.atoms[1].coordinates,4),
                                     *np.round(hbond.atoms[hbond_acceptor].coordinates,4)])
        structure["crystal"]["hbonds"] = structure_hbonds
            
        # Write data to files
        io_operations.write_structure_data(structure_data_file,structure["crystal"])
        io_operations.write_fragments_data(fragments_data_file,structure["crystal"],structure["fragments"])
        io_operations.write_contacts_data(contacts_data_file,structure["crystal"])
        io_operations.write_hbonds_data(hbonds_data_file,structure["crystal"])
        
                        
                        

    
