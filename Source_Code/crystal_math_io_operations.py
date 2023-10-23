import os
import json

def check_for_file(save_dir,filename):
    """
    Check if a file exists and if it does, ask the user whether to overwrite it or not.

    :param filename: str, the name of the file
    :return: file object
    """
    if os.path.exists(save_dir + filename):
        # If the file exists, ask for confirmation to overwrite
        user_input = input(f"File {filename} already exists. Do you want to overwrite it? (yes/no): ").lower()
        if user_input not in ['yes','y','YES','Y']:
            print(f"\tWARNING! Exiting without overwriting the file.\n\tNo data will be writen in file: {filename}")
            return None  # or manage the scenario where the user doesn't want to overwrite

    # If the file doesn't exist or if overwrite is confirmed, open and return the file object
    try:
        file_object = open(save_dir + filename, 'w')  # Open the file with writing mode, which will also create it if it doesn't exist
        return file_object
    except Exception as e:
        print(f"An error occurred: {e}")
        return None  
    
def write_cc_data(str_name,close_contacts,close_contacts_file):
    """ Writes the data for the close contacts """
    if close_contacts_file != None:
        for cc in close_contacts:
            atom_labels = [cc.atoms[0].label,cc.atoms[1].label]
            atom_symbols = [cc.atoms[0].atomic_symbol,cc.atoms[1].atomic_symbol]
            atom_labels.sort()
            atom_symbols.sort()
            close_contacts_file.write("{0:<8s} ".format(str_name))
            close_contacts_file.write("{0:<4s} {1:<4s} ".format(*atom_labels))
            close_contacts_file.write("{0:<4s} {1:<4s} ".format(*atom_symbols))
            close_contacts_file.write("{0:<16s}".format(cc.type.replace(' ','_')))
            close_contacts_file.write("{0:<7s}".format(str(cc.is_in_line_of_sight)))
            close_contacts_file.write("{0:8.4f} {1:8.4f}".format(cc.length,cc.strength))
            close_contacts_file.write("\n")

def write_fg_data(crystal_properties,fragment,inertia_eigenvectors,inertia_eigenvectors_planes_fractional,vectors_closest_to_perpendicular,fragment_atoms_pos,fragments_geometry_file):
    """ Writes the data for fragments geometry """
    if fragments_geometry_file != None:
        fragments_geometry_file.write("{0:<8s} ".format(crystal_properties["ID"]))
        fragments_geometry_file.write("{0:8.4f} {1:8.4f} {2:8.4f} ".format(*crystal_properties["scaled_cell_lengths"].tolist()))
        fragments_geometry_file.write("{0:8.2f} {1:8.2f} {2:8.2f} ".format(*crystal_properties["cell_angles"].tolist()))
        fragments_geometry_file.write("{0:<40s} ".format(fragment[4:]))
        fragments_geometry_file.write("{0:<4d} ".format(len(fragment_atoms_pos)))
        for vec in inertia_eigenvectors:
            fragments_geometry_file.write("{0:8.4f} {1:8.4f} {2:8.4f} ".format(*vec))
        for vec in inertia_eigenvectors_planes_fractional:
            fragments_geometry_file.write("{0:8.4f} {1:8.4f} {2:8.4f} ".format(*vec))
        for vec, ((w_i, angle_i), (w_j, angle_j)) in vectors_closest_to_perpendicular:
            fragments_geometry_file.write("{0:3d} {1:3d} {2:3d} {3:10.4f} {4:3d} {5:3d} {6:3d} {7:10.4f} ".format(*w_i,angle_i,*w_j,angle_j))
        for pos in fragment_atoms_pos:
            fragments_geometry_file.write("{0:8.4f} {1:8.4f} {2:8.4f} ".format(*pos))
        fragments_geometry_file.write("\n")
    
def write_hb_data(str_name,h_bonds,h_bonds_file):
    """ Writes the data for the close contacts """
    if h_bonds_file != None:
        for hb in h_bonds:
            atom_labels = [hb.atoms[0].label,hb.atoms[2].label]
            atom_symbols = [hb.atoms[0].atomic_symbol,hb.atoms[2].atomic_symbol]
            atom_labels.sort()
            atom_symbols.sort()
            atom_labels.insert(1,hb.atoms[1].label)
            atom_symbols.insert(1,hb.atoms[1].atomic_symbol)
            h_bonds_file.write("{0:<8s} ".format(str_name))
            h_bonds_file.write("{0:<4s} {1:<4s} {2:<4s} ".format(*atom_labels))
            h_bonds_file.write("{0:<4s} {1:<4s} {2:<4s} ".format(*atom_symbols))
            h_bonds_file.write("{0:<16s}".format(hb.type.replace(' ','_')))
            h_bonds_file.write("{0:<7s}".format(str(hb.is_in_line_of_sight)))
            h_bonds_file.write("{0:8.4f} {1:8.4f} {2:10.4f}".format(hb.length,hb.da_distance,hb.angle))
            h_bonds_file.write("\n")
            
def write_pi_data(str_name,fragment,inertia_eigenvectors_planes_fractional,intersections,plane_intersections_file):
    """ Writes the plane unit-cell vecrtices intersection data """
    if plane_intersections_file != None:
        plane_intersections_file.write("{0:<8s} ".format(str_name))
        plane_intersections_file.write("{0:<40s} ".format(fragment[4:]))
        for i, vec in enumerate(inertia_eigenvectors_planes_fractional):
            plane_intersections_file.write("{0:12.4f} {1:12.4f} {2:12.4f} ".format(*vec))
            for pos in intersections[i]:
                plane_intersections_file.write("{0:12.4f} {1:12.4f} {2:12.4f} ".format(*pos))
        plane_intersections_file.write("\n")
        
def write_str_data(crystal_properties,structure_data_file):
    """ Writes the general structure data """
    if structure_data_file != None:
        structure_data_file.write("{0:<8s} ".format(crystal_properties["ID"]))
        structure_data_file.write("{0:<8s} ".format(crystal_properties["space_group"]))
        structure_data_file.write("{0:6.2f} ".format(crystal_properties["z_crystal"]))
        structure_data_file.write("{0:6.2f} ".format(crystal_properties["z_prime"]))
        structure_data_file.write("{0:<10s} ".format(''.join(crystal_properties["species"])))
        structure_data_file.write("{0:8.4f} {1:8.4f} {2:8.4f} ".format(*crystal_properties["scaled_cell_lengths"].tolist()))
        structure_data_file.write("{0:8.4f} {1:8.4f} {2:8.4f} ".format(*crystal_properties["cell_lengths"].tolist()))
        structure_data_file.write("{0:8.2f} {1:8.2f} {2:8.2f} ".format(*crystal_properties["cell_angles"].tolist()))
        structure_data_file.write("{0:12.4f} ".format(crystal_properties["cell_volume"]))
        structure_data_file.write("{0:10.4f} ".format(crystal_properties["vdWFV"]))
        structure_data_file.write("{0:10.4f} ".format(crystal_properties["SAS"]))
        structure_data_file.write("\n")
                
            
    
    
    
