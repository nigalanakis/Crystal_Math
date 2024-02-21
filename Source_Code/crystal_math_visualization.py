import matplotlib.pyplot as plt
import numpy as np 
import os
from scipy.optimize import curve_fit

""" Define CCDC standardized colors """
ccdc_colors = {"C": (0.5703,0.5703,0.5703),
               "H": (1.0000,1.0000,1.0000),
               "N": (0.5625,0.5625,1.0000),
               "O": (0.9414,0.0000,0.0000),
               "S": (1.0000,0.7813,0.1914),
               "F": (0.7617,1.0000,0.0000),
               "Cl":(0.1250,0.9414,0.1250),
               "Br":(0.7471,0.5117,0.2383)}

def close_contacts_histogram(contacts_plots_dir, space_group, atom_pair, lengths_all, lengths_los, bins, colors, across_groups=False):
    plt.figure(figsize=(5, 3))

    # Plot histogram for all contacts
    n_all, bins_all, _ = plt.hist(lengths_all, bins=bins, color=colors['all_contacts'], alpha=0.7, label='All Contacts')

    # Plot histogram for line of sight contacts
    n_los, bins_los, _ = plt.hist(lengths_los, bins=bins, color=colors['line_of_sight'], alpha=0.7, label='Line of Sight Contacts')

    # Fit a Gaussian to the line of sight histogram
    if len(lengths_los) > 5:  # You can adjust this threshold
        bin_centers_los = 0.5 * (bins_los[1:] + bins_los[:-1])
        bin_centers_los = 0.5 * (bins_los[1:] + bins_los[:-1])

        try:
            popt_los, _ = curve_fit(
                gaussian,
                bin_centers_los,
                n_los,
                p0=[max(n_los), np.mean(lengths_los), np.std(lengths_los)],
                bounds=([0, min(lengths_los), 0], [np.inf, max(lengths_los), np.inf]),  # No negative values
                maxfev=5000  # Increase the maximum number of iterations
            )

            # Generate fitted curve data
            x_interval_for_fit = np.linspace(bin_centers_los[0], bin_centers_los[-1], 1000)
            plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt_los), color='red', linestyle='--')

            # Find and mark the maximum of the Gaussian fit
            max_fit_index = np.argmax(gaussian(x_interval_for_fit, *popt_los))
            max_fit_x = x_interval_for_fit[max_fit_index]
            max_fit_y = gaussian(x_interval_for_fit, *popt_los)[max_fit_index]
            plt.plot(max_fit_x, max_fit_y, 'ro', markersize=3)

            # Add a dotted line at the maximum
            plt.vlines(max_fit_x, 0, max_fit_y, colors='red', linestyles='dotted')
            plt.text(max_fit_x, max_fit_y, f'{max_fit_x:.2f} Å', color='red', ha='center', va='bottom')

        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            print("Gaussian fit did not converge for the line-of-sight data.")


    plt.xlabel('Contact Length (Å)')
    plt.ylabel('Frequency')
    title_suffix = "across all space groups" if across_groups else f"in {space_group}"
    plt.title(f'{atom_pair[0]}-{atom_pair[1]} Contacts {title_suffix}')
    
    # Improve the legend
    legend = plt.legend(frameon=1, shadow=True, loc='upper left')
    frame = legend.get_frame()
    frame.set_color('white')
    frame.set_edgecolor('black')

    # Fancy grid
    plt.grid(which='major', color='grey', linestyle='-', linewidth=0.5, alpha=0.7)
    # Minor gridlines
    plt.minorticks_on()  # Enable minor ticks
    plt.grid(which='minor', color='grey', linestyle=':', linewidth=0.5, alpha=0.5)
    
    # Directory structure and file naming
    if across_groups:
        directory = f'{contacts_plots_dir}/All_Space_Groups/'
        filename = f'histogram_{atom_pair[0]}-{atom_pair[1]}_All_SG.png'
    else:
        space_group_simple = space_group.replace('/','')
        directory = f'{contacts_plots_dir}/{space_group_simple}/'
        filename = f'histogram_{atom_pair[0]}-{atom_pair[1]}_{space_group_simple}.png'

    os.makedirs(directory, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{directory}/{filename}',dpi=300)
    plt.close()
    
def create_close_contacts_histograms(structure_data_file,contacts_data_file,contacts_plots_dir,bins):
    # Set the colors for the contacts histograms
    colors = {'all_contacts': '#1984c5', 'line_of_sight': '#c23728'}
    
    # Read space group information
    space_group_mapping = {}
    with open(structure_data_file, 'r') as file:
        for line in file:
            parts = line.split()
            structure_name = parts[0]
            space_group = parts[1]
            space_group_mapping[structure_name] = space_group
    
    # Prepare a nested dictionary for histograms
    histograms_by_space_group = {}
    histograms_for_pairs_across_groups = {}
    
    # Read contact data and create histograms
    with open(contacts_data_file, 'r') as file:
        for line in file:
            parts = line.split()
            structure_name = parts[0]
            atom_1_species = parts[3]
            atom_2_species = parts[4]
            is_in_line_of_sight = parts[6] == 'True'
            contact_length = float(parts[7])
    
            # Skip if the structure is not in the space group mapping
            if structure_name not in space_group_mapping:
                continue
    
            # Determine space group for the current structure
            space_group = space_group_mapping[structure_name]
    
            # Generate atom pair tuple, sorted to maintain consistency
            atom_pair = tuple(sorted([atom_1_species, atom_2_species]))
    
            # Initialize histograms for this space group and atom pair if not already present
            if space_group not in histograms_by_space_group:
                histograms_by_space_group[space_group] = {}
            if atom_pair not in histograms_by_space_group[space_group]:
                histograms_by_space_group[space_group][atom_pair] = {'all': [], 'los': []}
    
            # Append the contact length to the appropriate histogram list
            histograms_by_space_group[space_group][atom_pair]['all'].append(contact_length)
            if is_in_line_of_sight:
                histograms_by_space_group[space_group][atom_pair]['los'].append(contact_length)
    
            # Collect data for all space groups for this atom pair
            if atom_pair not in histograms_for_pairs_across_groups:
                histograms_for_pairs_across_groups[atom_pair] = {'all': [], 'los': []}
            histograms_for_pairs_across_groups[atom_pair]['all'].append(contact_length)
            if is_in_line_of_sight:
                histograms_for_pairs_across_groups[atom_pair]['los'].append(contact_length)
                
    # Loop over space groups and atom pairs to create histograms
    for space_group, atom_pairs in histograms_by_space_group.items():
        for atom_pair, contact_lengths in atom_pairs.items():
            close_contacts_histogram(
                contacts_plots_dir,
                space_group,
                atom_pair,
                contact_lengths['all'],
                contact_lengths['los'],
                bins,
                colors
            )
    
    # Now create histograms for each atom pair across all space groups
    for atom_pair, contact_lengths in histograms_for_pairs_across_groups.items():
        close_contacts_histogram(
            contacts_plots_dir,
            "AllSpaceGroups",
            atom_pair,
            contact_lengths['all'],
            contact_lengths['los'],
            bins,
            colors,
            across_groups=True
        )
        
def create_h_bonds_histograms(structure_data_file,h_bonds_data_file,h_bonds_plots_dir,measurement,bins):
    # Set the colors for the contacts histograms
    colors = {'all_contacts': '#1984c5', 'line_of_sight': '#c23728'}
    
    # Read space group information
    space_group_mapping = {}
    with open(structure_data_file, 'r') as file:
        for line in file:
            parts = line.split()
            structure_name = parts[0]
            space_group = parts[1]
            space_group_mapping[structure_name] = space_group
    
    # Prepare a nested dictionary for histograms
    histograms_by_space_group = {}
    histograms_for_triplets_across_groups = {}
    
    # Read contact data and create histograms
    with open(h_bonds_data_file, 'r') as file:
        for line in file:
            parts = line.split()
            structure_name = parts[0]
            atom_1_species = parts[4]
            atom_2_species = parts[5]
            atom_3_species = parts[6]
            is_in_line_of_sight = parts[8] == 'True'
            value = float(parts[measurement + 9])
            
            # Skip if the structure is not in the space group mapping
            if structure_name not in space_group_mapping:
                continue
    
            # Determine space group for the current structure
            space_group = space_group_mapping[structure_name]
    
            # Generate atom pair and triplet tuple, sorted to maintain consistency
            atom_pair = tuple(sorted([atom_1_species, atom_3_species]))
            atom_triplet = (atom_pair[0], atom_2_species, atom_pair[1])

            # Initialize histograms for this space group and atom triplet if not already present
            if space_group not in histograms_by_space_group:
                histograms_by_space_group[space_group] = {}
            if atom_triplet not in histograms_by_space_group[space_group]:
                histograms_by_space_group[space_group][atom_triplet] = {'all': [], 'los': []}
    
            # Append the contact length to the appropriate histogram list
            histograms_by_space_group[space_group][atom_triplet]['all'].append(value)
            if is_in_line_of_sight:
                histograms_by_space_group[space_group][atom_triplet]['los'].append(value)
    
            # Collect data for all space groups for this atom pair
            if atom_triplet not in histograms_for_triplets_across_groups:
                histograms_for_triplets_across_groups[atom_triplet] = {'all': [], 'los': []}
            histograms_for_triplets_across_groups[atom_triplet]['all'].append(value)
            if is_in_line_of_sight:
                histograms_for_triplets_across_groups[atom_triplet]['los'].append(value)
         
    # Loop over space groups and atom pairs to create histograms
    labels = ['Bond length', 'D-A distance', 'Bond angle']
    
    for space_group, atom_triplets in histograms_by_space_group.items():
        for atom_triplet, measurements in atom_triplets.items():
            h_bonds_histogram(
                h_bonds_plots_dir,
                space_group,
                atom_triplet,
                measurements['all'],
                measurements['los'],
                bins,
                colors,
                labels[measurement]
            )
    
    # Now create histograms for each atom pair across all space groups
    for atom_triplet, measurements in histograms_for_triplets_across_groups.items():
        h_bonds_histogram(
            h_bonds_plots_dir,
            "AllSpaceGroups",
            atom_triplet,
            measurements['all'],
            measurements['los'],
            bins,
            colors,
            labels[measurement],
            across_groups=True
        )
            
def fragment_eigenvectors_plot(fragment_atoms_species, fragment_atoms_mass, fragment_atoms_bv, inertia_eigenvectors, point=[0,0,0], limit=2):
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    axes_colors = ['blue','green','red']
    
    # Plot point.
    x, y, z = point
    ax.plot(x, y, z, marker='o', markersize=5, color='black')
    
    # Plot pnincipal axes of inertia and the respective perpendicular planes
    for axis, vector in enumerate(inertia_eigenvectors):
        d = -np.sum(vector * point)
        
        # Create a meshgrid:
        delta = 2
        xlim = point[0] - delta, point[0] + delta
        ylim = point[1] - delta, point[1] + delta
        xx, yy = np.meshgrid(np.arange(*xlim), np.arange(*ylim))
        
        # Solving the equation above for z:
        # z = -(a*x + b*y +d) / c
        zz = -(vector[0] * xx + vector[1] * yy + d) / vector[2]
        
        # Plot vector.
        ax.plot_surface(xx, yy, zz, alpha=0.25, color=axes_colors[axis])
        dx, dy, dz = delta * vector
        ax.quiver(0, 0, 0, dx, dy, dz, arrow_length_ratio=0.15, linewidth=1, color=axes_colors[axis])
        
        # Enforce equal axis aspects so that the normal also appears to be normal.
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        zlim = point[2] - delta, point[2] + delta
        ax.set_zlim(*zlim)
        
        # Label axes.
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        
    # Plot the atoms    
    atom_colors = [ccdc_colors[species] for species in fragment_atoms_species]
    ax.scatter(fragment_atoms_bv[:,0],fragment_atoms_bv[:,1],fragment_atoms_bv[:,2],color=atom_colors,s=3*fragment_atoms_mass,edgecolors='black',linewidth=0.5,alpha=1)
    ax.set_xlim(-limit,limit)
    ax.set_ylim(-limit,limit)
    ax.set_zlim(-limit,limit)
    
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 2 / stddev)**2)


def h_bonds_histogram(h_bonds_plots_dir, space_group, atom_triplet, measurements_all, measurements_los, bins, colors, label, across_groups=False):
    plt.figure(figsize=(5, 3))
    
    # Plot histogram for all bonds
    n_all, bins_all, _ = plt.hist(measurements_all, bins=bins, color=colors['all_contacts'], alpha=0.7, label='All Bonds')

    # Plot histogram for line of sight bonds
    n_los, bins_los, _ = plt.hist(measurements_los, bins=bins, color=colors['line_of_sight'], alpha=0.7, label='Line of Sight Bonds')

    # Fit a Gaussian to the line of sight histogram only if we have enough data points
    if len(measurements_los) > 5:  # You can adjust this threshold
        bin_centers_los = 0.5 * (bins_los[1:] + bins_los[:-1])
        try:
            popt_los, _ = curve_fit(
                gaussian,
                bin_centers_los,
                n_los,
                p0=[max(n_los), np.mean(measurements_los), np.std(measurements_los)],
                bounds=([0, min(measurements_los), 0], [np.inf, max(measurements_los), np.inf]),  # No negative values
                maxfev=5000  # Increase the maximum number of iterations
            )

            # Generate fitted curve data
            x_interval_for_fit = np.linspace(bin_centers_los[0], bin_centers_los[-1], 1000)
            plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt_los), color='red', linestyle='--')

            # Find and mark the maximum of the Gaussian fit
            max_fit_index = np.argmax(gaussian(x_interval_for_fit, *popt_los))
            max_fit_x = x_interval_for_fit[max_fit_index]
            max_fit_y = gaussian(x_interval_for_fit, *popt_los)[max_fit_index]
            plt.plot(max_fit_x, max_fit_y, 'ro', markersize=3)

            # Add a dotted line at the maximum
            plt.vlines(max_fit_x, 0, max_fit_y, colors='red', linestyles='dotted')
            plt.text(max_fit_x, max_fit_y, f'{max_fit_x:.2f}', color='red', ha='center', va='bottom')

        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            print(f"Gaussian fit did not converge for the line-of-sight data.")
            
    plt.xlabel(f'{label} (Å)')
    plt.ylabel('Frequency')
    atom_triplet_str = '-'.join(atom_triplet)
    title_suffix = "across All Space Groups" if across_groups else f"in {space_group}"
    plt.title(f'Histogram of {label} for {atom_triplet_str} {title_suffix}')
    
    # Improve the legend
    legend = plt.legend(frameon=1, shadow=True, loc='upper left')
    frame = legend.get_frame()
    frame.set_color('white')
    frame.set_edgecolor('black')

    # Fancy grid
    plt.grid(which='major', color='grey', linestyle='-', linewidth=0.5, alpha=0.7)
    # Minor gridlines
    plt.minorticks_on()  # Enable minor ticks
    plt.grid(which='minor', color='grey', linestyle=':', linewidth=0.5, alpha=0.5)
    
    # Directory structure and file naming
    file_label = label.replace(' ','_')
    if across_groups:
        directory = f'{h_bonds_plots_dir}/All_Space_Groups/'
        filename = f'histogram_{file_label}_{atom_triplet_str}_All_SG.png'
    else:
        space_group_simple = space_group.replace('/','')
        directory = f'{h_bonds_plots_dir}/{space_group_simple}/'
        filename = f'histogram_{file_label}_{atom_triplet_str}_{space_group_simple}.png'

    os.makedirs(directory, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{directory}/{filename}',dpi=300)
    plt.close()
    

