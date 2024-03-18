import os 

def check_for_file(save_dir,filename):
    """
    Check if a file exists and if it does, ask the user whether to overwrite it or not.

    Parameters
    ----------
    filename : str
        The name of the file
    
    Returns
    -------
    file object
    """
    if os.path.exists(save_dir + filename):
        # If the file exists, ask for confirmation to overwrite
        user_input = input(f'File {filename} already exists. Do you want to overwrite it? (yes/no): ').lower()
        if user_input not in ['yes','y','YES','Y']:
            print(f'\tWARNING! Exiting without overwriting the file.\n\tNo data will be writen in file: {filename}')
            return None  # or manage the scenario where the user doesn't want to overwrite

    # If the file doesn't exist or if overwrite is confirmed, open and return the file object
    try:
        file_object = open(save_dir + filename, 'w')  # Open the file with writing mode, which will also create it if it doesn't exist
        return file_object
    except Exception as e:
        print(f'An error occurred: {e}')
        return None 

def create_file_headers(structure_data_file,fragments_data_file,contacts_data_file,hbonds_data_file):
    if structure_data_file != None:
        formatted_data = (f"{'str_id':<8s} "
                           f"{'sg':>8s} {'Z':>6s} {'Z_pr':>6s} "
                           f"{'formula':>30s} {'species':>10s} "
                           f"{'a_sc':>8s} {'b_sc':>8s} {'c_sc':>8s} "
                           f"{'a':>8s} {'b':>8s} {'c':>8s} "
                           f"{'alpha':>8s} {'beta':>8s} {'gamma':>8s} "
                           f"{'volume':>12s} {'density':>12s} "
                           f"{'vdWFV':>10s} {'SAS':>10s} "
                           f"{'E_tot':>12s} {'E_el':>12s} "
                           f"{'E_vdW':>12s} {'E_vdW_at':>12s} {'E_vdW_rep':>12s} "
                           f"{'E_hb':>12s} {'E_hb_at':>12s} {'E_hb_rep':>12s}\n")
        structure_data_file.write(formatted_data)
    if fragments_data_file != None:
        formatted_data = (f"{'str_id':<8s} "
                          f"{'fragment':>30s} "
                          f"{'x':>8s} {'y':>8s} {'z':>8s} "
                          f"{'u':>8s} {'v':>8s} {'w':>8s} "
                          f"{'e1_x':>8s} {'e1_y':>8s} {'e1_z':>8s} "
                          f"{'e2_x':>8s} {'e2_y':>8s} {'e2_z':>8s} "
                          f"{'e3_x':>8s} {'e3_y':>8s} {'e3_z':>8s} "
                          f"{'d1':>8s} {'d2':>8s} {'d3':>8s} "
                          f"{'e1_u':>8s} {'e1_v':>8s} {'e1_w':>8s} "
                          f"{'e2_u':>8s} {'e2_v':>8s} {'e2_w':>8s} "
                          f"{'e3_u':>8s} {'e3_v':>8s} {'e3_w':>8s} "
                          f"{'W11_u':>6s} {'W11_v':>6s} {'W11_w':>6s} {'ang_11':>6s} "
                          f"{'W12_u':>6s} {'W12_v':>6s} {'W12_w':>6s} {'ang_12':>6s} "
                          f"{'W21_u':>6s} {'W21_v':>6s} {'W21_w':>6s} {'ang_21':>6s} "
                          f"{'W22_u':>6s} {'W22_v':>6s} {'W22_w':>6s} {'ang_22':>6s} "
                          f"{'W31_u':>6s} {'W31_v':>6s} {'W31_w':>6s} {'ang_31':>6s} "
                          f"{'W32_u':>6s} {'W32_v':>6s} {'W32_w':>6s} {'ang_32':>6s} "
                          f"{'n_at':>6s} "
                          f"{'at_x':>8s} {'at_y':>8s} {'at_z':>8s} "
                          f"{'at_u':>8s} {'at_v':>8s} {'at_w':>8s} {'dzzp_min':>8s}\n")
        fragments_data_file.write(formatted_data)
    if contacts_data_file != None:
        formatted_data = (f"{'str_id':<8s} "
                          f"{'label1':>6s} {'label2':>6s} {'spec1':>6s} {'spec2':>6s} {'hbond':>6s} "
                          f"{'central_fragment':>30s} {'contact_fragment':>30s} {'length':>8s} {'in_los':>6s} "
                          f"{'x1':>8s} {'y1':>8s} {'z1':>8s} "
                          f"{'x2':>8s} {'y2':>8s} {'z2':>8s} "
                          f"{'bvx1':>8s} {'bvy1':>8s} {'bvz1':>8s} "
                          f"{'bvx2':>8s} {'bvy2':>8s} {'bvz2':>8s} "
                          f"{'bvx1_ref':>8s} {'bvy1_ref':>8s} {'bvz1_ref':>8s} "
                          f"{'bvx2_ref':>8s} {'bvy2_ref':>8s} {'bvz2_ref':>8s} "
                          f"{'r2':>8s} {'theta2':>8s} {'phi2':>8s}\n")
        contacts_data_file.write(formatted_data)
    if hbonds_data_file != None:
        formatted_data = (f"{'str_id':<8s} "
                          f"{'labelD':>6s} {'labelH':>6s} {'labelA':>6s} "
                          f"{'specD':>6s} {'specH':>6s} {'specA':>6s} "
                          f"{'length':>8s} {'DA_dis':>8s} {'angle':>8s} "
                          f"{'in_los':>6s} "
                          f"{'xD':>8s} {'yD':>8s} {'zD':>8s} "
                          f"{'xH':>8s} {'yH':>8s} {'zH':>8s} "
                          f"{'xA':>8s} {'yA':>8s} {'zA':>8s}\n")
        hbonds_data_file.write(formatted_data)
        
def write_structure_data(structure_data_file,crystal_properties):
    """ Writes the general structure data """
    # formula = ''.join(str(crystal_properties['formula']).split(' '))
    if structure_data_file != None:
        formatted_data = (f"{crystal_properties['ID']:<8s} "
                          f"{crystal_properties['space_group']:>8s} "
                          f"{crystal_properties['z_crystal']:6.2f} "
                          f"{crystal_properties['z_prime']:6.2f} "
                          f"{''.join(str(crystal_properties['formula']).split(' ')):>30s} "
                          f"{''.join(crystal_properties['species']):>10s} "
                          f"{' '.join(f'{length:8.4f}' for length in crystal_properties['scaled_cell_lengths'].tolist())} "
                          f"{' '.join(f'{length:8.4f}' for length in crystal_properties['cell_lengths'].tolist())} "
                          f"{' '.join(f'{angle:8.2f}' for angle in crystal_properties['cell_angles'].tolist())} "
                          f"{crystal_properties['cell_volume']:12.4f} "
                          f"{crystal_properties['cell_density']:12.4f} "
                          f"{crystal_properties['vdWFV']:10.4f} "
                          f"{crystal_properties['SAS']:10.4f} ")
        energy_properties = ['total', 'electrostatic', 'vdW', 'vdW_attraction', 'vdW_repulsion', 'h-bond', 'h-bond_attraction', 'h-bond_repulsion']
        for property_name in energy_properties:
            formatted_data += f"{crystal_properties['lattice_energy'][property_name]:12.4e} "
        formatted_data += '\n'
        structure_data_file.write(formatted_data)
        
def write_fragments_data(fragments_data_file,crystal_properties,fragments):
    """ Writes the fragment data """
    if fragments_data_file != None:
        for key in fragments:
            fragment = fragments[key]
            formatted_data = (
                f"{crystal_properties['ID']:<8s} "
                f"{key[4:]:>30s} "
                f"{' '.join(f'{coord:8.4f}' for coord in fragment['coordinates_c'].tolist())} "
                f"{' '.join(f'{coord:8.4f}' for coord in fragment['coordinates_f'].tolist())} "
                f"{' '.join(f'{val:8.4f}' for row in fragment['rotation_matrix'] for val in row)} "
                f"{' '.join(f'{dist:8.4f}' for dist in fragment['principal_inertia_planes_distances_to_cell_points'])} "
                f"{' '.join(f'{val:8.4f}' for plane in fragment['principal_inertia_planes_f'] for val in plane)} "
                f"{' '.join(f'{vec[0]:6d} {vec[1]:6d} {vec[2]:6d} {angle:6.2f}' for row in fragment['n_max_vectors'] for vec, angle in row[1])} "
                f"{len(fragment['atoms']):6d} ")
            fragments_data_file.write(formatted_data)
            for (x, y, z), (u, v, w), d in zip(fragment["atoms_coordinates_c"], fragment["atoms_coordinates_f"], fragment["minimum_atom_distances_to_zzp_planes"]):
                fragments_data_file.write(f'{x:8.4f} {y:8.4f} {z:8.4f} {u:8.4f} {v:8.4f} {w:8.4f} {d:8.4f} ')
            fragments_data_file.write('\n')
            
def write_contacts_data(contacts_data_file,crystal_properties):
    """ Write the contacts data """
    if contacts_data_file != None:
        for contact in crystal_properties["close_contacts"]:
            formatted_data = (f"{crystal_properties['ID']:<8s} "
                              + ' '.join(f"{item:>6s}" for item in map(str, contact[:5])) + ' '
                              + f"{contact[5]:>30s} {contact[6]:>30s} {contact[7]:8.4f} {str(contact[8]):>6s} "
                              + ' '.join(f"{item:8.4f}" for item in contact[9:27]) + ' ' 
                              + f"{contact[27]:8.4f} {contact[28]:8.4f} {contact[29]:8.2f}\n" )
            contacts_data_file.write(formatted_data)
            
def write_hbonds_data(hbonds_data_file,crystal_properties):
    """ Write the contacts data """        
    if hbonds_data_file != None:
        for hbond in crystal_properties["hbonds"]:
            formatted_data = (f"{crystal_properties['ID']:<8s} "
                              + ' '.join(f"{hbond[i]:>6s}" for i in range(6)) + ' '
                              + ' '.join(f"{hbond[i]:8.4f}" for i in range(6, 9))  
                              + f" {str(hbond[9]):>6s} "  
                              + ' '.join(f"{item:8.4f}" for item in hbond[10:19]) + '\n')
            hbonds_data_file.write(formatted_data)
                
        
    
    
    