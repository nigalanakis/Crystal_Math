import ast 
import numpy as np
import os
import sys

from ccdc import io
from ccdc.descriptors import MolecularDescriptors as MD
from datetime import datetime
from time import process_time as timer   

import crystal_math_fragment_properties as fragment_properties
import crystal_math_io_operations as cmio
import crystal_math_maths as maths
import crystal_math_structure_properties as structure_properties
import crystal_math_visualization as visualization
 
def main(input_file):
    # Load execution parameters 
    with open(input_file) as f:
        data = f.read()

    input_parameters = ast.literal_eval(data)
    
    # Open files to write data
    close_contacts_file = cmio.check_for_file(input_parameters["save_directory"],input_parameters["contacts_data_file"])
    fragment_geometry_file = cmio.check_for_file(input_parameters["save_directory"],input_parameters["fragments_geometry_data_file"])
    h_bonds_file = cmio.check_for_file(input_parameters["save_directory"],input_parameters["h-bonds_data_file"])
    plane_intersections_file = cmio.check_for_file(input_parameters["save_directory"],input_parameters["plane_intersection_data_file"])
    structure_data_file = cmio.check_for_file(input_parameters["save_directory"],input_parameters["structure_data_file"])

    # Create the list of structures to analyze 
    if input_parameters["structures_list"][0] == "CSD":
        csd_entries = io.EntryReader("CSD")
        if input_parameters["structures_list"][1] == "CSD":
            structures_list = io.EntryReader("CSD")
        else: 
            structures_list = input_parameters["structures_list"][1]
    elif input_parameters["structures_list"][0] == "cif":
        structures_dir = input_parameters["structures_list"][1]
        structures_list = input_parameters["structures_list"][2]
        
    # Get the framgents list
    fragments_list = fragment_properties.fragment_list(input_parameters["fragments_input_file"])

    # Loop over all structures
    n_structures = 0
    for str_name in structures_list:
        # Generate crystal and molecules
        if input_parameters["structures_list"][1] == "CSD":
            crystal = str_name.crystal 
            molecule = str_name.molecule
        
        if input_parameters["structures_list"][0] == "CSD" and input_parameters["structures_list"][1] != "CSD":
            entry = csd_entries.entry(str_name)
            crystal = entry.crystal 
            molecule = entry.molecule
            
        if input_parameters["structures_list"][0] == "cif":
            crystal = io.CrystalReader(structures_dir + str_name)
            crystal = crystal[0]
            
            molecule = io.MoleculeReader(structures_dir + str_name)
            molecule = molecule[0]  
        
        if input_parameters["center_molecule"]:
            crystal.centre_molecule() # Move molecule inside unit cell 
        
        # Discard structures with zon-integer Z prime value 
        if crystal.z_prime not in input_parameters["target_z_prime_values"]:
            continue

        # Discard structures with unwanted space group 
        Proceed = True
        if input_parameters["target_space_groups"] != [] and crystal.spacegroup_symbol not in input_parameters["target_space_groups"]:
            continue
        
        # Assign unknow bond types, add missing hydrogens and assign 
        # partial charges to atoms
        try:
            molecule.assign_bond_types()
            molecule.add_hydrogens()
            molecule.assign_partial_charges()
        except Exception:
            continue
        
        # Generate atoms
        try:
            atoms = molecule.atoms 
        except Exception:
            continue
        
        # Discard structures with no atoms in the crystal
        if len(atoms) == 0:
            continue
            
        # Discard structures with out-of-range molecular weight
        Proceed = True
        for component in molecule.components:
            if component.molecular_weight > input_parameters["molecule_weight_limit"]:
                Proceed = False
                break
        if not Proceed:
            continue
        
        # Get crystal, molecule and atom properties
        crystal_properties = structure_properties.crystal_properties(crystal)
        atom_properties, molecule_properties = structure_properties.atom_and_molecule_properties(crystal,
                                                                                                 molecule,
                                                                                                 atoms)
        
        # Discard structures with unwanted atomic species
        Proceed = True
        if input_parameters["target_species"] != []:
            for s in crystal_properties["species"]:
                if s not in input_parameters["target_species"]:
                    Proceed = False
                    break
                    
        if not Proceed:
            continue

        fragments = fragment_properties.structure_fragments(fragments_list,
                                                            atom_properties["label"],
                                                            atom_properties["species"],
                                                            molecule)
        
        # Add fragments for full components 
        if input_parameters["add_full_component"]:
            for i, component in enumerate(molecule.components):
                component_atoms_labels = [at.label for at in component.atoms]
                component_atoms = [atom_properties["label"].index(at.label) for at in component.atoms]

                fragments["component_" + str(i + 1)] = {"atoms": component_atoms, "atom_labels": component_atoms_labels}
        
        # Discard structures with none of the desired substructures 
        if not bool(fragments):
            continue
    
        print('Alalyzing structure ' + crystal_properties["ID"])
        n_structures += 1
        # Loop over all fragments to calculate the fragment orientation
        for fragment in fragments:
            # Get fragment properties
            fragment_atoms = fragments[fragment]["atoms"] # Get the atom list for the fragment
            fragment_atoms_species = fragments[fragment]["atom_species"] # Get the atom list for the fragment
            fragment_atoms_mass = atom_properties["mass"][fragment_atoms] # Get the mass of the atoms in the fragment
            fragment_atoms_pos = atom_properties["coordinates_c"][fragment_atoms] # Get the physical positions of the atoms in the fragment
            fragment_atoms_ref_sfc = np.array(fragments[fragment]["ref_pos"]) # Get the reference positions of the atoms for the fragment in the body fixed frame
            fragment_com = maths.center_of_mass(fragment_atoms_mass,fragment_atoms_pos) # Calculate the center of mass for the fragment
            fragment_atoms_bv = fragment_atoms_pos - fragment_com # Calculate the bond vectors for the fragment
             
            # Get the list of atoms that are used for the aligmnent
            if fragments[fragment]["atoms_to_align"] == "all":
                fragment_atoms_to_align  = range(len(fragment_atoms))
            else:
                fragment_atoms_to_align = fragments[fragment]["atoms_to_align"]
            
            # Get the rotation matrix
            rotation_matrix = maths.kabsch_rotation_matrix(fragment_atoms_ref_sfc[fragment_atoms_to_align], 
                                                           fragment_atoms_bv[fragment_atoms_to_align])
            
            # Filter unwanted fragments in case of identical smarts representation
            if fragment[4:] in input_parameters["fragments_to_check_alignment"]:
                fragment_atoms_ref_bv, rmsd = maths.align_structures(rotation_matrix, 
                                                                     fragment_atoms_ref_sfc[fragment_atoms_to_align], 
                                                                     fragment_atoms_bv[fragment_atoms_to_align])
                if rmsd > input_parameters["alignment_tolerance"]:
                    continue 
            
            # Extrach the eigevectors
            inertia_eigenvectors = rotation_matrix

            # Calculate the normalized vectors perpendicular to the 
            # principal inertia planes in the crystallographic coordinates 
            # system
            inertia_eigenvectors_planes_fractional = np.dot(inertia_eigenvectors, crystal_properties['h-matrix'])
            inertia_eigenvectors_planes_fractional = inertia_eigenvectors_planes_fractional / np.linalg.norm(inertia_eigenvectors_planes_fractional, axis=1, keepdims=True)
            
            ###############################################################
            ### IMPORTANT NOTE                                          ###
            ###############################################################
            ### The principal axes of inertia are the rows of the       ###
            ### rotation_matrix!                                        ### 
            ###############################################################
            ### The principal planes of inertia in crystallographic     ###
            ### coordinates are the rows of                             ###
            ### inertia_eigenvectors_planes_fractional                  ###
            ############################################################### 
            
            # Visualize eigenvectors
            if input_parameters["visualize_eigenvectors"]:
                visualization.fragment_eigenvectors_plot(fragment_atoms_species, 
                                                         fragment_atoms_mass, 
                                                         fragment_atoms_bv, 
                                                         inertia_eigenvectors)

            # Identify for each eigenvector the proposed vectors that are
            # closest to be perpendicular and the respective angle
            vectors_closest_to_perpendicular = maths.find_vectors_closest_to_perpendicular(inertia_eigenvectors_planes_fractional, 
                                                                                           input_parameters["proposed_vectors_n_max"])

            # Calculate the intersection points between the principal
            # planes of inertia in crystallographic coordinates and the 
            # vertices of the unit cell
            intersections = maths.plane_cube_intersections(inertia_eigenvectors_planes_fractional, 
                                                           np.dot(fragment_com, crystal_properties['t-matrix']).T)
            
            # Write fragments geometry data to file
            cmio.write_fg_data(crystal_properties,
                               fragment,
                               inertia_eigenvectors.tolist(),
                               inertia_eigenvectors_planes_fractional.tolist(),
                               vectors_closest_to_perpendicular,
                               fragment_atoms_pos.tolist(),
                               fragment_geometry_file)

            # Write plane intersections data to file
            cmio.write_pi_data(crystal_properties["ID"],
                               fragment,
                               inertia_eigenvectors_planes_fractional.tolist(),
                               intersections,
                               plane_intersections_file)
            
        # Write structure data to file
        cmio.write_str_data(crystal_properties,
                            structure_data_file)
        
        # Write close contacts data to file
        cmio.write_cc_data(crystal_properties["ID"],
                           crystal_properties["close_contacts"],
                           close_contacts_file)
                
        # Write h-bonds data to file
        cmio.write_hb_data(crystal_properties["ID"],
                           crystal_properties["h-bonds"],
                           h_bonds_file)

    return(n_structures)
                
if __name__ == "__main__":
    input_file = "input_files/input_csd_data_extraction.txt"
    
    now = datetime.now()
    print("Process started at ", now.strftime("%Y-%m-%d %H:%M:%S"))

    start = timer()
    n_structures = main(input_file)
    
    cpu_time = timer() - start
    hours, minutes, seconds = maths.convert_seconds(cpu_time)
    now = datetime.now()
    print("Process completed at ", now.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Number of structures analyzed: {n_structures}")
    print(f"Total computation time: {hours}h {minutes}m {seconds:.2f}s")
