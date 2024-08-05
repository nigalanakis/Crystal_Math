import copy 
import json 
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from create_reference_fragments import create_reference_fragments
from utilities import format_lists, convert_to_json

# Enable LaTeX rendering
plt.rcParams['text.usetex'] = True

# Use the Computer Modern Unicode fonts and amsmath
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # for mathematical fonts
plt.rcParams['font.family'] = 'serif'

# Define CCDC color mapping for common elements
def atoms_color_map():
    '''
    Sets atom colors based on the atomic species
    
    Parameters
    ----------
    None.
    
    Returns
    -------
    atom_colors : dict
        A disctionary with the colors of different atomic species.
    '''
    atom_colors = {
        'H': '#bebebe',    
        'C': '#777777',  
        'N': '#a1a1ff',    
        'O': '#cb0000',   
        'F': '#c6ff12',   
        'Cl': '#44ff3a',   
        'Br': '#be823c',   
        'I': '#990099',    
        'S': '#ffcb35',    
        'P': '#ff9421'
    }
    
    return atom_colors

def get_dataframe(data,var_family,var_list,sg,filters):
    '''
    Gets the dataframe for the histogram/plot

    Parameters
    ----------
    data : dict
        A dictionary with the structure variables data.
    var_family : str 
        The family of the variable(s).
    var_list : list
        A list of the variables to beused for the histogram/plot.
    sg : str
        The space group filter for the data.
    filters : dict
        a dictionary with the user defined filters for the variables.

    Returns
    -------
    df : pandas dataframe
        The dataframe for the data to be used for the histogram/plot.
    '''
    if sg == 'all':
        # Group values across all space groups
        df_list = []
        for space_group in data[var_family]:
            for values in zip(*(data[var_family][space_group][var] for var in var_list)):
                df_list.append(dict(zip(var_list, values)))
        df = pd.DataFrame(df_list, columns = var_list)
    else:
        # Get the dataframe for the required variables (plot variable + filter variables)
        df = pd.DataFrame({var: data[var_family][sg][var] for var in var_list})

    # Apply filters if necesary
    if filters != None:
        for filter_variable, values in zip(filters.keys(), filters.values()):
            for value in values:
                df = df[df[filter_variable] == value]
    return df

def get_variable_limits(df,var_list):
    '''
    Calculates the variable limits

    Parameters 
    ----------
    df : pandas dataframe
        The dataframe that is used for the histogram/plot.
    var_list : list
        A list of the variables to beused for the histogram/plot.

    Returns
    -------
    limits : list
        A list with the limits for the variables to be used for the histogram/plot
    ''' 
    # Get the list of available variables
    with open('../source_data/variables.json') as f:
        variables = json.load(f)

    var_range = [max(df[var]) - min(df[var]) for var in var_list]
    limits = [[min(df[var]) - 0.1*var_range[i], max(df[var]) + 0.1*var_range[i]] for i, var in enumerate(var_list)]
    for i, var in enumerate(var_list):
        if var == 'cc_length':
            limits[i] = [0,4]
        elif variables[var]['position_symmetry'][2]:
            limits[i] = [0,1]
    return limits

def set_figure_title(output_format, var_name, suffix):
    '''  
    Sets the figure title

    Parameters
    ----------
    output_format : str 
        The format of the output file ('html' or 'png').
    var_name : str 
        The name of the variable in html or LaTeX format.
    suffix : str
        The space group suffix for the plot.

    Returns
    -------
    figtitle : str
        The figure title
    '''
    if output_format == 'png':
        suffix = '$' + suffix + '$'
        figtitle = ' '.join([var_name, ''.join(['(',suffix.replace('_',' ').replace('21', '2_1'),')'])])
        for text in [' (\\AA)', ' (deg)', ' (\\AA$^3$)', ' (gr/cm$^3)', ' (\\%)', ' (kJ/mol)']:
            figtitle = figtitle.replace(text, '')
    elif output_format == 'html':
        figtitle = ' '.join([var_name, ''.join(['(',suffix.replace('21', '2<sub>1</sub>'),')']).replace('_',' ')])
        for text in [' (Ang.)', ' (deg)', ' (Ang.<sup>3</sup>)', ' (gr/cm<sup>3</sup>)', ' (%)', ' (kJ/mol)']:
            figtitle = figtitle.replace(text, '')
    return figtitle

def set_html_text_replacements(text):
    '''
    Replaces html text if figures/titles with the appropriate for rendering.

    Parameters
    ----------
    text : str
        The input text.

    Returns
    -------
    text : str
        The updated text.
    '''
    replacements = {
        'Ang.': 'Å'
    }
    for key, value in zip(replacements.keys(),replacements.values()):
        text = text.replace(key, value)
    return text 

def get_density_ranges(density,percentiles,percentiles_list):
    '''
    Calculate the density ranges for the scatter plots

    Pamareters
    ----------
    density : obj
        The calculated KDE density.
    percentiles : list
        A list with the percentiles
    percentiles_list : list
        The user defined list with the percentiles

    Returns
    -------
    density_ranges : list
        A list with the calculated density ranges
    '''
    density_ranges = []
    density_ranges_names = []
    for i in range(len(percentiles) + 1):
        if i == 0:
            density_ranges.append(density < percentiles[i])
            density_ranges_names.append(f'[0,{percentiles_list[i]})')
        elif i == len(percentiles):
            density_ranges.append(density >= percentiles[i - 1])
            density_ranges_names.append(f'[{percentiles_list[i - 1]},100]')
        else:
            density_ranges.append((density >= percentiles[i - 1]) & (density < percentiles[i]))
            density_ranges_names.append(f'[{percentiles_list[i - 1]},{percentiles_list[i]})')
    return density_ranges, density_ranges_names

def get_density_color_settings(steps,density_ranges):
    '''
    Sets the colors for the scatter plots density ranges

    Parameters
    ----------
    steps : int
        The number of density ranges
    density_ranges : list
        A list with the calculated density ranges

    Returns
    -------
    colors : list
        A list with the color for each density range.
    sizes : list
        A list with the size for each density range.
    opacities : list
        A list with the opacity for each density range.
    '''
    # Set the colors
    jet = plt.get_cmap('jet')
    colors = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b, a in [jet(pos) for pos in steps]]
    sizes = 4 * (steps + 1)
    opacities = 0.5 + np.zeros(len(density_ranges))
    opacities[-1] = 1.0
    return colors, sizes, opacities

def create_scatter_data(output_format,markers,x,y,z,n_dimensions,density_ranges=None,density_ranges_names=None):
    scatter_data = []
    if output_format == 'png':
        if density_ranges != None:
            for i in range(len(density_ranges)):
                current_markers = {
                    'shape': {'matplotlib': markers['shape']['matplotlib']},
                    'face_color': markers['face_color'][i],
                    'edge_color': markers['edge_color'],
                    'size': markers['size'][i],
                    'opacity': markers['opacity'][i]}
                if n_dimensions == 2:
                    scatter_data.append([current_markers, x[density_ranges[i]], y[density_ranges[i]], None, n_dimensions, density_ranges_names[i]])
                else: 
                    scatter_data.append([current_markers, x[density_ranges[i]], y[density_ranges[i]], z[density_ranges[i]], n_dimensions, density_ranges_names[i]])
        else:
            scatter_data.append([markers, x, y, z, n_dimensions, None])
    elif output_format == 'html':
        if density_ranges != None:
            for i in range(len(density_ranges)):
                if n_dimensions == 2:
                    trace = go.Scatter(
                        x=x[density_ranges[i]],
                        y=y[density_ranges[i]],
                        mode='markers',
                        marker=dict(
                            symbol=markers['shape']['plotly'],
                            color=markers['face_color'][i],
                            size=markers['size'][i],
                            opacity=markers['opacity'][i],
                            line=dict(color=markers['edge_color'], width=0.5)
                        ),
                        name=density_ranges_names[i]
                    )
                else:
                    trace = go.Scatter3d(
                        x=x[density_ranges[i]],
                        y=y[density_ranges[i]],
                        z=z[density_ranges[i]],
                        mode='markers',
                        marker=dict(
                            symbol=markers['shape']['plotly'],
                            color=markers['face_color'][i],
                            size=markers['size'][i],
                            opacity=markers['opacity'][i],
                            line=dict(color=markers['edge_color'], width=0.5)
                        ),
                        name=density_ranges_names[i]
                    )
                scatter_data.append(trace)
        else:
            if n_dimensions == 2:
                trace = go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    marker=dict(
                        symbol=markers['shape']['plotly'],
                        color=markers['face_color'],
                        size=10.0,
                        opacity=1.0,
                        line=dict(color=markers['edge_color'], width=0.5)
                    ),
                    name='all'
                )
            else:
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode='markers',
                    marker=dict(
                        symbol=markers['shape']['plotly'],
                        color=markers['face_color'],
                        size=10.0,
                        opacity=1.0,
                        line=dict(color=markers['edge_color'], width=0.5)
                    ),
                    name='all'
                )
            scatter_data = [trace]
    return scatter_data
    
def generate_histogram(output_format,figsize,colors,df,var,var_name,limits,calculate_density,fit_curve,suffix,save_directory,save_fig):
    '''
    Generates a histogram for the 'png' or 'html' format plots.
    
    Parameters
    ----------
    output_format : str 
        The format of the output file ('html' or 'png').
    figsize : list
        The user defined size for the 'png' figures. 
    colors : dict
        A dictionary with the user defined colors for the different elements in the histogram/plot.
    df : pandas dataframe
        The data for the plot.
    var : str
        The variable for the histogram.
    var_name : str 
        The LaTeX/html name for the variable that is used for the x-axis label and figure title.
    limits : list
        A list with the limits for the variables to be used for the histogram/plot
    calculate_density : bool
        A boolean that determines if the density will be plotted instead of the occurences.
    fit_curve : bool
        A boolean that controls if the algorithm will fit a curve to the data.
    suffix : str
        The space group suffix for the plot.
    save_directory : str
        The directory to save the plots.
    save_fig : boolean
        Controls if the figures will be saved in the save_directory folder. 
    '''
    # Get the list of available variables
    with open('../source_data/variables.json') as f:
        variables = json.load(f)

    # Set the data range
    data_range = limits[0][1]-limits[0][0]

    if output_format == 'png':
        fig, ax = plt.subplots(1,1,figsize=figsize)
        
    if calculate_density or fit_curve:
        if output_format == 'png':
            # Calculate the number of bins
            num_bins = int((limits[0][1] - limits[0][0]) / (data_range / 100))
            ax.hist(df[var], bins=num_bins, range=(limits[0][0], limits[0][1]), color=colors['bar'], density=True)
            ax.hist(df[var], bins=num_bins, range=(limits[0][0], limits[0][1]), color=colors['line'], histtype='step', density=True)
        if output_format == 'html':
            fig = go.Figure(data=[go.Histogram(x=df[var],
                                               histnorm='probability density',
                                               xbins=dict(start=limits[0][0], end=limits[0][1], size=data_range/100),
                                               marker=dict(color=colors['bar']))])
    else:
        if output_format == 'png':
            num_bins = int((limits[0][1] - limits[0][0]) / (data_range / 100))
            ax.hist(df[var], bins=num_bins, range=(limits[0][0], limits[0][1]), color=colors['bar'])
            ax.hist(df[var], bins=num_bins, range=(limits[0][0], limits[0][1]), color=colors['line'], histtype='step')
        elif output_format == 'html':
            fig = go.Figure(data=[go.Histogram(x=df[var],
                                               xbins=dict(start=limits[0][0], end=limits[0][1], size=data_range/100),
                                               marker=dict(color=colors['bar']))])

    # Fit a KDE
    if fit_curve and len(df[var]) >= 3:
        try:
            # Calculate KDE density
            if variables[var]['position_symmetry'][2]:
                period = 1.0
                extended_df = np.concatenate([df[var], df[var] + period, df[var] - period])
                kde = gaussian_kde(extended_df)
            else:
                kde = gaussian_kde(df[var])
    
            # Define evaluation points
            x_values = np.linspace(min(df[var]), max(df[var]), 1000)
            y_values = kde(x_values)
            
            if variables[var]['position_symmetry'][2]:
                normalization_factor = np.trapz(y_values, x_values)
                y_values /= normalization_factor

            # Find the maximum of the Gaussian fit
            max_fit_index = np.argmax(y_values)
            max_fit_x = x_values[max_fit_index]
            max_fit_y = y_values[max_fit_index]
            
            # Add the KDE curve to the plot
            if output_format == 'png':                
                # Add a dotted vertical line, a point and a label at the maximum
                ax.text(max_fit_x, max_fit_y, f'{max_fit_x:.2f}', color=colors['point'], ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.50, edgecolor='none'))
                ax.vlines(max_fit_x, 0, max_fit_y, colors=colors['point'], linestyles='dotted')
                ax.scatter(max_fit_x, max_fit_y, marker='o', color=colors['point'], s=5)

                # Plot the fit line
                ax.plot(x_values, y_values, label='KDE Curve', color=colors['fit'])
            if output_format == 'html':
                # Adding a point at the maximum
                fig.add_trace(go.Scatter(
                    x=[max_fit_x],
                    y=[max_fit_y],
                    mode='markers',  # Use 'markers' to display only points
                    marker=dict(
                        color=colors['point'],  # Set the color of the point
                        size=10  # Set the size of the marker
                    ),
                    name='Point Marker'  # Name of the trace, can be used in legends
                ))

                # Add a dotted line at the maximum
                fig.add_shape(
                    type='line',
                    x0=max_fit_x, x1=max_fit_x, y0=0, y1=max_fit_y,
                    line=dict(
                        color=colors['point'],
                        width=2,
                        dash='dot'
                             )
                )
                
                # Add text annotation at the position of the maximum
                fig.add_annotation(
                    x=max_fit_x, y=max_fit_y,
                    text=f'{max_fit_x:.2f}',
                    showarrow=False,
                    font=dict(
                        color=colors['point'], 
                        family='CMU Serif', 
                        size=20
                        ),
                    xanchor='center', 
                    yanchor='bottom'
                )

                fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='KDE Curve', line=dict(color=colors['fit'])))
        except np.linalg.LinAlgError:
            print(f'Low dimension data for {var}. Unable to fit curve.')

    # Set the figure title
    figtitle = set_figure_title(output_format, var_name, suffix)

    # Update layout for better visualization
    y_axis_title = 'Frequency' if calculate_density or fit_curve else 'Occurences'
        
    if output_format == 'png':
        ax.set_xlabel(var_name)
        ax.set_ylabel(y_axis_title)
        ax.set_title(figtitle)

        # Improve the legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend = plt.legend(frameon=1, shadow=True, loc='upper left')
            frame = legend.get_frame()
            frame.set_color('white')
            frame.set_edgecolor('black')
    
        # Fancy grid
        plt.grid(which='major', color='grey', linestyle='-', linewidth=0.5, alpha=0.7)
        # Minor gridlines
        plt.minorticks_on()  # Enable minor ticks
        plt.grid(which='minor', color='grey', linestyle=':', linewidth=0.5, alpha=0.5)
        
        plt.tight_layout()
    elif output_format == 'html':
        # Replace text in x-axis label
        x_axis_title = set_html_text_replacements(var_name)
            
        fig.update_layout(
            title=figtitle,
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            bargap=0.2,  # Gap between bars
            bargroupgap=0.1,  # Gap between groups of bars,
            xaxis=dict(
                range=limits[0]), 
            template='none',
            font=dict(
                family='CMU Serif',
                size=20,
                color='Black'
            )
        )

    # Save the plot
    if save_fig:
        figname = '_'.join(['Hist', var, suffix.replace('/', '')]) 
        if output_format == 'png':
            plt.savefig(save_directory + figname + '.png', dpi=300)
            plt.close()
        if output_format == 'html':
            fig.write_html(save_directory + figname + '.html', include_mathjax = 'cdn')
          
    return
    
def create_histogram(input_parameters,save_directory,data):
    '''
    Creates histograms based on the input variables.
    
    Parameters
    ----------
    input_parameters : dict
        A dictionary with the user defined input parameters.
    save_directory : str
        The directory to save the plots.
    data : dict
        A dictionary with the user defined filtered data for the plots.
    
    Returns
    -------
    None
    '''
    # Get the list of available variables
    with open('../source_data/variables.json') as f:
        variables = json.load(f)

    # Set if the figures will be saved or not and the output type
    save_figures = input_parameters['general_analysis_options']['save_figures']
    output_format = input_parameters['general_analysis_options']['output_format']

    # Set the colors
    colors = input_parameters['histograms_options']['colors']

    # Set the figure size for png format
    if 'png' in output_format:
        figsize = input_parameters['general_analysis_options']['figure_size']

    # Loop over the variables
    for var_group, filters, calculate_density, fit_curve in input_parameters['histograms_options']['variables']:
        # Get the plot variables, the plot variables family and their names for the plots.
        var_list = var_group
        var_family = variables[var_list[0]]['family']
        var_list_html_names = [variables[var]['html_name'] for var in var_list]
        var_list_latex_names = [variables[var]['latex_name'] for var in var_list]

        # Determine if we ask to filter variables based on a property
        if filters != None:     
            var_list.extend([var for var in filters])

        # Get the space group list
        space_group_list = ['all'] + [space_group for space_group in data[var_family]] if input_parameters['general_analysis_options']['individual_space_groups'] else ['all']

        # Loop over space groups to create histograms
        for sg in space_group_list:
            # Get the data for the plot
            df = get_dataframe(data,var_family,var_list,sg,filters)

            # Check if the data frame is empty
            if df.empty:
                continue

            # Get the range of values based on the values across all space groups
            # The range is determine by the min and max value for the variable with the exception of the contact length and fractional atomic coordinates
            if sg == 'all':
                limits = get_variable_limits(df,[var_list[0]])

            # Generate the histogram
            if 'png' in input_parameters['general_analysis_options']['output_format']:
                generate_histogram('png',figsize,colors,df,var_list[0],var_list_latex_names[0],limits,calculate_density,fit_curve,sg,save_directory,save_figures)
            if 'html' in input_parameters['general_analysis_options']['output_format']:
                generate_histogram('html',None,colors,df,var_list[0],var_list_html_names[0],limits,calculate_density,fit_curve,sg,save_directory,save_figures)
    
    return 
    
def generate_scatter_plot(n_dimensions,output_format,figsize,markers,df,filters,var_list,var_list_names,percentiles_list,suffix,save_directory,save_fig):
    '''
    Generates 2D or 3D scatter plots in 'png' or 'html' format.
    
    Parameters
    ----------
    n_dimensions : int
        The number of fdimensions for the scatter plot (2 or3).
    output_format : str 
        The format of the output file ('html' or 'png').
    figsize : list
        The user defined size for the 'png' figures. 
    markers : dict
        A dictionary with the user defined marker appearance details.
    df : pandas dataframe
        The data for the plot.
    filters : dict
        A dictionary with the filters for the specific set of variables.
    var_list : list
        The variables for the scatter plot.
    var_list_names : list
        The LaTeX/html names for the variables that are used for scatter plot axes labels and title.
    percentiles_list : list
        A list of the KDE densities percentiles to be calculated.
    suffix : str
        The space group suffix for the plot.
    save_directory : str
        The directory to save the plots.
    save_fig : boolean
        Controls if the figures will be saved in the save_directory folder. 
    '''
    # if creating a png image, initialize figure 
    if output_format == 'png':
        data = []
        if n_dimensions == 2:
            fig, ax = plt.subplots(1,1,figsize=figsize)
            # plt.scatter(df[var_list[0]],df[var_list[1]],marker=marker,facecolor=facecolor,edgecolor=edgecolor,alpha=alpha)
        else:    
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            # ax.scatter(df[var_list[0]],df[var_list[1]],df[var_list[2]],marker=marker,facecolor=facecolor,edgecolor=edgecolor,alpha=alpha)
        
    # Set the axes
    x = df[var_list[0]]
    y = df[var_list[1]]
    z = df[var_list[2]] if n_dimensions == 3 else None

    if percentiles_list != None and len(df) >= n_dimensions:
        # If we require density scatter plot, do not create simple scatter plot
        simple_scatter = False

        # Prepare data for kde density calculation
        data_vstack = np.vstack([x, y] if n_dimensions == 2 else [x, y, z])

        # Calculate densities
        try:
            # Perform PCA to reduce dimensionality if necessary
            pca = PCA(n_components=n_dimensions)
            data_vstack_pca = pca.fit_transform(data_vstack.T).T
            
            kde = gaussian_kde(data_vstack_pca)
            density = kde(data_vstack_pca)
            
            # Assign percentiles
            percentiles = np.percentile(density, percentiles_list)
        
            # Get density ranges
            density_ranges, density_ranges_names = get_density_ranges(density, percentiles, percentiles_list)
    
            # Get color settings
            steps = np.linspace(0, 1, len(density_ranges))
            colors, sizes, opacities = get_density_color_settings(steps, density_ranges)
    
            # Set the marker colors
            markers['face_color'] = colors
            markers['opacity'] = opacities
            markers['size'] = sizes
    
            # Create scatter data for the plot
            data = create_scatter_data(output_format, markers, x, y, z, n_dimensions, density_ranges, density_ranges_names)
        except:
            # If KDE fails due to singular matrix, activate simple scatter plot
            simple_scatter = True
    else:
        # If we do not require density scatter plot, activate simple scatter plot
        simple_scatter = True

    if simple_scatter:
        data = create_scatter_data(output_format, markers, x, y, z, n_dimensions)

    # If plotting contacts on the reference frame, also plot the atoms in reference fragment.
    if var_list[:3] in [['cc_contact_atom_ref_bv_' + coor for coor in ['x', 'y', 'z']],
                    ['cc_central_atom_ref_bv_' + coor for coor in ['x', 'y', 'z']]]:
        atom_colors = atoms_color_map()
        fragment_list = create_reference_fragments()
        for fragment in fragment_list:
            if fragment in filters['cc_central_atom_fragment']:
                fragment_atom_coors = pd.DataFrame(fragment_list[fragment]['coordinates_sf'],columns=['x','y','z'])
                fragment_atom_species = fragment_list[fragment]['species']
                fragment_atom_colors = [atom_colors[species] for species in fragment_atom_species]

                xf = fragment_atom_coors['x']
                yf = fragment_atom_coors['y']
                zf = fragment_atom_coors['z']

                if output_format == 'png':
                    fragment_markers = {
                        'shape': {"matplotlib": 'o', "plotly": "circle"},
                        'face_color': fragment_atom_colors,
                        'edge_color': 'black',
                        'size': 100,
                        'opacity': 1.0}
                    data.append([fragment_markers, xf, yf, zf, n_dimensions, ['fragment']])
                elif output_format == 'html':
                    data.append(go.Scatter3d(
                        x=xf,
                        y=yf,
                        z=zf,
                        mode='markers',
                        marker=dict(
                            color=fragment_atom_colors,
                            size=12.0,
                            opacity=1.0,
                            line=dict(color='black', width=0.5)
                            ),
                            name='fragment'
                        )
                    )

    # Set the figure title
    joined_var_names = ' vs '.join(var_list_names)
    figtitle = set_figure_title(output_format, joined_var_names, suffix)
        
    # Set the figure layout
    if output_format == 'png':
        ax.set_xlabel(var_list_names[0])
        ax.set_ylabel(var_list_names[1])
        if n_dimensions == 3:
            ax.set_zlabel(var_list_names[2])
        ax.set_title(figtitle)
        plt.tight_layout()
    elif output_format == 'html':
        layout = go.Layout(
            title=dict(text=figtitle, x=0.5, xanchor='center'),
            scene=dict(
                xaxis=dict(title=var_list_names[0]),
                yaxis=dict(title=var_list_names[1]),
                zaxis=dict(title=var_list_names[2]) if n_dimensions == 3 else None
            )
        ) if n_dimensions == 3 else go.Layout(
            title=dict(text=figtitle, x=0.5, xanchor='center'),
            xaxis_title=var_list_names[0],
            yaxis_title=var_list_names[1]
        )
    
    # Create plot
    if output_format == 'png':
        legend_labels = []
        for marker, x, y, z, n_dimensions, names in data:
            if names != None:
                legend_labels.append(names)
            if n_dimensions == 2:
                ax.scatter(x, y, marker=marker['shape']['matplotlib'],s=marker['size'],facecolor=marker['face_color'],edgecolor=marker['edge_color'],alpha=marker['opacity'])
            else:
                ax.scatter(x, y, z, marker=marker['shape']['matplotlib'],s=marker['size'],facecolor=marker['face_color'],edgecolor=marker['edge_color'],alpha=marker['opacity'])
        if legend_labels != []:
            ax.legend(labels=legend_labels)
    elif output_format == 'html':
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(
            template="none",
            font=dict(
                family="CMU Serif",
                size=20,
                color="Black"
            )
        )
    
    # Save the plot as an HTML file
    if save_fig:
        common_prefix = os.path.commonprefix(var_list[:n_dimensions])
        suffixes = [item[len(common_prefix):] for item in var_list[:n_dimensions]]
        joined_var_names = common_prefix + "_vs_".join(suffixes)
        
        figname = '_'.join([f'{n_dimensions}D_sp', joined_var_names, suffix.replace("/", "")]) 
        if output_format == 'png':
            plt.savefig(save_directory + figname + '.png', dpi=300)
            plt.close()
        elif output_format == 'html':
            fig.write_html(save_directory + figname + '.html')
    return

def create_scatter_plot(n_dimensions,input_parameters,save_directory,data):
    '''
    Creates histograms based on the input variables.
    
    Parameters
    ----------
    n_dimensions : int
        The number of fdimensions for the scatter plot (2 or3).
    input_parameters : dict
        A dictionary with the user defined input parameters.
    save_directory : str
        The directory to save the plots.
    data : dict
        A dictionary with the user defined filtered data for the plots.
    
    Returns
    -------
    None
    '''
    # Get the list of available variables
    with open('../source_data/variables.json') as f:
        variables = json.load(f)

    # Set if the figures will be saved or not and the output type
    save_figures = input_parameters['general_analysis_options']['save_figures']
    output_format = input_parameters['general_analysis_options']['output_format']

    # Set the key for the input parameters scatter options
    scatter_options_key = f'{n_dimensions}D_scatter_plots_options'
    
    # Set the marker details
    markers = input_parameters[scatter_options_key]['markers']

    # Set the figure size for png format
    if 'png' in output_format:
        figsize = input_parameters['general_analysis_options']['figure_size']

    # Loop over the variables
    for var_group, filters, densities in input_parameters[scatter_options_key]['variables']:
        # Get the plot variables, the plot variables family and their names for the plots.
        var_list = var_group
        var_family = variables[var_list[0]]['family']
        var_list_html_names = [variables[var]['html_name'] for var in var_list]
        var_list_latex_names = [variables[var]['latex_name'] for var in var_list]

        # Determine if we ask to filter variables based on a property
        if filters != None:     
            var_list.extend([var for var in filters])
        
        # Get the space group list
        space_group_list = ['all'] + [space_group for space_group in data[var_family]] if input_parameters['general_analysis_options']['individual_space_groups'] else ['all']

        # Loop over space groups to create plots
        for sg in space_group_list:
            # Get the data for the plot
            df = get_dataframe(data,var_family,var_list,sg,filters)

            # Check if the data frame is empty
            if df.empty:
                continue

            # Get the range of values based on the values across all space groups
            # The range is determine by the min and max value for the variable with the exception of the contact length and fractional atomic coordinates
            if sg == 'all':
                limits = get_variable_limits(df,[var_list[0]])

            # Create the scatter plots
            if 'png' in input_parameters['general_analysis_options']['output_format']:
                generate_scatter_plot(n_dimensions,'png',figsize,copy.deepcopy(markers),df,filters,var_list,var_list_latex_names,densities,sg,save_directory,save_figures)
            
            if 'html' in input_parameters['general_analysis_options']['output_format']:
                generate_scatter_plot(n_dimensions,'html',None,copy.deepcopy(markers),df,filters,var_list,var_list_html_names,densities,sg,save_directory,save_figures)
    return

def create_correlation_map(input_parameters,save_directory,data):
    '''
    Creates correlation maps for the selected variables 
    
    Parameters
    ----------
    input_parameters : dict
        A dictionary with the user defined input parameters.
    save_directory : str
        The directory to save the plots.
    data : dict
        A dictionary with the user defined filtered data for the plots.

    Returns
    -------
    '''
    # Get the list of available variables
    with open('../source_data/variables.json') as f:
        variables = json.load(f)

    # Set if the figures will be saved or not and the output type
    save_figures = input_parameters['general_analysis_options']['save_figures']

    # Set the minimum acceptable data size and the method
    min_data_size = input_parameters['correlation_options']['min_data_size']
    correlation_method = input_parameters['correlation_options']['method']

    for i_group, (var_group, filters) in enumerate(input_parameters['correlation_options']['variables']):
        var_list = var_group 
        var_family = variables[var_list[0]]['family']
        var_names = [variables[var]['latex_name'] for var in var_list]
        var_names_no_units = []
        for name in var_names: 
            for text in [' (\\AA)', ' (deg)', ' (\\AA$^3$)', ' (gr/cm$^3)', ' (\\%)', ' (kJ/mol)']:
                name = name.replace(text, '')
            var_names_no_units.append(name)

        # Get the space group list
        space_group_list = ['all'] + [space_group for space_group in data[var_family]] if input_parameters['general_analysis_options']['individual_space_groups'] else ['all']
        
        # Determine if we ask to filter variables based on a property
        if filters != None:     
            var_list.extend([var for var in filters])

        # Loop over space groups to create histograms
        for sg in space_group_list:
            # Get the data for the map
            df = get_dataframe(data,var_family,var_list,sg,filters)
            
            # Check if the data frame is empty
            if df.empty or len(df) < min_data_size:
                continue

            correlation_matrix = df.corr(method=correlation_method)

            plt.figure(figsize=(len(var_list)+2, len(var_list)))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, xticklabels=var_names_no_units, yticklabels=var_names_no_units)
            plt.title(set_figure_title('png', f'Correlation Heatmap {i_group + 1} ({correlation_method.capitalize()})', sg) + f' ({len(df)})')
            plt.tight_layout()
            
            # Save the plot as an HTML file
            if save_figures:
                figname = '_'.join([f'Correlation_Heatmap_{i_group + 1}_{correlation_method.capitalize()}', sg.replace("/", "")]) 
                plt.savefig(save_directory + figname + '.png', dpi=300)
            plt.close()
    return

import copy
def fragment_to_ellipsoid(input_parameters,save_directory,data):
    # Get the list of available variables
    with open('../source_data/variables.json') as f:
        variables = json.load(f)
        
    # Loop over the fragments
    for filters in input_parameters['fragments_ellipsoids_options']['filters']:
        # Set the variables, the variable family and the names
        var_group = [f'cc_central_atom_ref_bv_{coor}' for coor in ['x','y','z']] + [f'cc_contact_atom_ref_bv_{coor}' for coor in ['x','y','z']]
        var_list = [f'cc_central_atom_ref_bv_{coor}' for coor in ['x','y','z']] + [f'cc_contact_atom_ref_bv_{coor}' for coor in ['x','y','z']]
        var_family = 'contact_atom'
        var_list_html_names = [variables[var]['html_name'] for var in var_list]
        var_list_latex_names = [variables[var]['latex_name'] for var in var_list]

        # Extend variables list to filter variables based on a property
        var_list.extend([var for var in filters])

        # Get the space group list
        space_group_list = ['all'] + [space_group for space_group in data[var_family]] if input_parameters['general_analysis_options']['individual_space_groups'] else ['all']
   
        # Loop over space groups to create plots
        for sg in space_group_list:
            # Get the data for the plot
            df = get_dataframe(data,var_family,var_list,sg,filters).drop_duplicates(subset=var_group)

            # Check if the data frame is empty
            if df.empty:
                continue

            # Create data points for the surface of the ellipsoid
            df['dx'] = df['cc_contact_atom_ref_bv_x'] - df['cc_central_atom_ref_bv_x']
            df['dy'] = df['cc_contact_atom_ref_bv_y'] - df['cc_central_atom_ref_bv_y']
            df['dz'] = df['cc_contact_atom_ref_bv_z'] - df['cc_central_atom_ref_bv_z']




            print(var_list)
            print(var_family)
            print(sg)
            print(filters)
            print(df.head())
            print()
        print()

    print('So far so good')
    return