Post extraction analysis
========================
This section outlines the default post-extraction analysis tools. 
The purpose of this tool is to perform qualitative and quantitative analysis of the structure, fragment, contact and hydrogen bond data for the selected group of structures.
The tool is designed to create scatter plots for pairs of parameters and histograms for the parameters extracted during the data extraction process.

For the scatter plots, the algorithm calculates the correlation coefficients for the selected set of variables, while for the histograms, it offers the option to fit distributions to the selected data, and report the characteristics of the fitted curve.

The Data Analysis Input File
----------------------------
The first step is to modify the ``input_data_analysis.txt`` file based on the required criteria. The general format of the file and descriptions of each parameter are as follows:

Input File Format
^^^^^^^^^^^^^^^^^
The configuration should be specified in JSON format as shown below:

.. code-block:: json

    {
         "plots_directory": "../csd_db_analysis/visualize/",
         "data_directory": "../csd_db_analysis/db_data/",
         "data_prefix": "homomolecular",
         "folder": "contacts_carboxylic-acid_carboxylic-acid_OH_hb_Zprime_1",
         "figure_size": [5,3.75],
         "save_figs": false,
         "data_filters": {
             "space_group": {
                 "is_active": false,
                 "type": "single",
                 "values": ["P21/c","P21/n"],
                 "operator": "or",
                 "refine_data": false
             }, 
             "z_crystal": {
                 "is_active": false,
                 "type": "single",
                 "values": [4,8],
                 "operator": "or",
                 "refine_data": false
             }, 
             "z_prime": {
                 "is_active": true,
                 "type": "single",
                 "values": [1],
                 "operator": "or",
                 "refine_data": false
             },
             "species": {
                 "is_active": false,
                 "type": "multiple",
                 "values": ["C","H","N","O"],
                 "operator": "or",
                 "refine_data": false
             },
             "fragments": {
                 "is_active": true,
                 "type": "multiple",
                 "values": [
                     "carboxylic_acid",
                     // ...
                     ],
                 "operator": "and",
                 "refine_data": true
             },
             "contact_pairs": {
                 "is_active": true,
                 "type": "multiple_lists",
                 "values": [
                     ["O","H","hbond",true],
                     // ...
                     ],
                 "operator": "or",
                 "refine_data": true
             },
             "contact_central_fragments": {
                 "is_active": true,
                 "type": "multiple_lists",
                 "values": [
                     ["carboxylic_acid","hbond",true]
                     // ...
                     ],
                 "operator": "or",
                 "refine_data": true
             },
             "contact_fragment_pairs": {
                 "is_active": true,
                 "type": "multiple_lists",
                 "values": [
                     ["carboxylic_acid","carboxylic_acid","hbond",true],
                     // ...
                     ],
                 "operator": "and",
                 "refine_data": true
             }
         },
         "plot_data_options": {
            "individual_space_groups_plots": true,
            "interactive": true,
            "percentiles": [[10,25,50,75,90],true,true,true],
            "2D_scatter": [
                ["cell_length_b_sc","cell_length_c_sc",null],
                // ...
        		],
            "2D_scatter_marker": "o",
            "2D_scatter_facecolor": "whitesmoke",
            "2D_scatter_edgecolor": "black",
            "2D_scatter_opacity": 1.0,
            "3D_scatter": [
                ["cc_contact_atom_ref_bv_x","cc_contact_atom_ref_bv_y","cc_contact_atom_ref_bv_z",null],
                // ...
                ],
            "3D_scatter_marker": "o",
            "3D_scatter_facecolor": "whitesmoke",
            "3D_scatter_edgecolor": "black",
            "3D_scatter_opacity": 1.0,
            "histogram": [
                ["cc_length",null,false],
                // ...
                ],
            "histogram_density": false,
            "titles": false
         }
    }


Key Descriptions
^^^^^^^^^^^^^^^^
- ``plots_directory``
    Specifies the directory where plots will be saved. Using the default option is recommended.
- ``data_directory``
    The directory where the extracted data is stored. It must match the ``"save_directory"`` specified in the ``input_data_extraction.json`` file.
- ``data_prefix``
    A prefix applied to output files to facilitate their identification. This must be consistent with the ``"data_prefix"`` in the ``input_data_extraction.json`` file.
- ``figure_size``
    Defines the dimensions of exported figures in inches, formatted as :math:`(W \times H)`. The default Matplotlib size is :math:`(6.4 \times 4.8)`. To place two figures side by side in a 12-inch wide document using an 11pt font, the optimal size is :math:`(5.0 \times 3.75)`. Adjust dimensions according to your document's specific requirements.
- ``data_filters``
    Details for filtering structures for the analysis. Structures can be filtered based on: 

    - **Space group**
        The space group of the structure.
    - :math:`Z` **value**
        The total number of molecules in the unit cell (Number of symmetry operations) :math:`\times` (Number of molecules in the asymmetric unit).
    - :math:`Z^{\prime}` **value**
        The number of molecules in the asymmetric unit.
    - **Atomic species**
        The different atomic species found in the structure.
    - **Fragments**
        The different fragments found in the structure.
    - **Contact atomic pairs**    
        The different atomic pairs found for the contacts in the structure.
    - **Contact central fragments**
        The different central fragments for the contacts in the structure.
    - **Contact fragment pairs**
        The different fragment pairs found for the contacts in the structure.
    
    Each filter has 5 options:

    - ``is_active``
        Set to ``true`` to activate the filter. Setting to ``false`` will deactivate the filter.
    - ``type``
        The type of the filter. The available options are 
    
        - ``single``
            A structure is characterized by a single specific value for the variable (for example the space group).
        - ``multiple``
            A structure is characterized by a list of values for the specific variable (for example the atomic species in the structure).
        - ``multiple_list``
            A structure is characterized by a list of values for the specific variable, but each value is now a list (for example the contact pairs in the structure, where each contact pair is characterized by the species of the cetnral atom, the species of the contact atom, the type of the contact and a boolean that states if the contact is in line of sight).
    
    - ``values``
        A list (or a list of lists) for the allowed values.
    - ``operator``
        The available options are
    
        - ``"or"``
            The filter will check for structures that have **any** of the declared values,
        - ``"and"``
            The filter will check for structures that have **all** the declared values,
        
    - ``refine_data``
        Set to ``true`` to refine the data for all the components in the structure based on the values of the filter. 

- ``plot_data_options`` 
    Details the plotting options:

    - ``individual_space_groups_plots``
        Set to ``true`` to create plots across all space groups and for each pace group sepaately.

    - ``interactive``
        Set to ``true`` to create interactive `*.html`` plots with the plotly package. (Currently this is the only option supported. Currently developing a routine to generate publication-ready ``*.png`` plots).

    - ``percentiles``
        The options to calculate the kde density for the 2D and 3D scatter plots. The format for the values includes a list of integerss (of floats) representing the desired percentiles followed by 3 booleans. Each boolean activates the creation of the lowest percentine (in the example the 10%), the middle percentines (25%, 50%, 75%), and the top percentile (90%). For the interactive ``*.html``` plots, it is recommended to set all options to ``true`` as the interactive plots allow to toggle on/off the different percentiles. For static ``*.png`` images, the booleans should be adjusted to include the desired percentiles in the plots. 

    - ``2D_scatter``/``3D_scatter``
        A list of the requested 2D/3D scatter plots to be generated. Each entry has the format ``[variable_1, variable_2, group_variable]``/``[variable_1, variable_2, variable_3, group_variable]``. The ``variable_1``, ``variable_2`` and ``variable_3`` are the variables used for the scatter plots. The entry ``group_variable`` declares the variable to group data and plot them separately based on the values of the group variable. Setting ``group_variable`` to ``null`` generates a single plot for the full set of selected data. The group variable can take different values depending on the nature of  ``variable_1``, ``variable_2``, ``variable_3``.

    - ``2D_scatter_marker``/``3D_scatter_marker``
        The marker for the data points (static images only). For the available options please refer to the `official matplotlib documentation <https://matplotlib.org/stable/api/markers_api.html>`_. 
    
    - ``2D_scatter_facecolor``/``3D_scatter_facecolor``
        The marker face color for the data points (static images only). For the available options please refer to the `official matplotlib documentation <https://matplotlib.org/stable/gallery/color/named_colors.html>`_. 
    
    - ``2D_scatter_edgecolor``/``3D_scatter_edgecolor``
        The marker edge color for the data points (static images only). For the available options please refer to the `official matplotlib documentation <https://matplotlib.org/stable/gallery/color/named_colors.html>`_. 
    
    - ``2D_scatter_opacity``/``3D_scatter_opacity``
        The marker opacity for the data points (static images only). Can take a value in the range :math:`[0,1]`.

    - ``histogram``
        A list of the requested histograms to be generated. Each entry has the format ``[variable, group_variable, fit_kde_curve]``. The ``group_variable`` works in a similar was as for the 2D/3D scatter plots. the     ``fit_kde_curve`` can be set to ``true`` when we require to fit a kde curve to the histogram data.  

    - ``histogram_density``
        Setting to ``false`` will plot on the ``y`` axis the occurences. Setting to ``true`` will plot the frequency. 

List of available variables
---------------------------

The available variables are included in the file ``variables.json`` located in the ``source_data`` folder. Currently, the algorithm supports 127 different variables:

- ``str_id``
- ``space_group``
- ``z_crystal``
- ``z_prime``
- ``formula``
- ``species``
- ``cell_length_a``
- ``cell_length_b``
- ``cell_length_c``
- ``cell_length_a_sc``
- ``cell_length_b_sc``
- ``cell_length_c_sc``
- ``cell_angle_alpha``
- ``cell_angle_beta``
- ``cell_angle_gamma``
- ``cell_volume``
- ``cell_density``
- ``vdWFV``
- ``SAS``
- ``E_tot``
- ``E_el``
- ``E_vdW``
- ``E_vdW_at``
- ``E_vdW_rep``
- ``E_hb``
- ``E_hb_at``
- ``E_hb_rep``
- ``cc_length``
- ``cc_type``
- ``cc_is_in_los``
- ``cc_central_atom_species``
- ``cc_central_atom_fragment``
- ``cc_central_atom_x``
- ``cc_central_atom_y``
- ``cc_central_atom_z``
- ``cc_central_atom_u``
- ``cc_central_atom_v``
- ``cc_central_atom_w``
- ``cc_central_atom_bv_x``
- ``cc_central_atom_bv_y``
- ``cc_central_atom_bv_z``
- ``cc_central_atom_ref_bv_x``
- ``cc_central_atom_ref_bv_y``
- ``cc_central_atom_ref_bv_z``
- ``cc_contact_atom_species``
- ``cc_contact_atom_fragment``
- ``cc_contact_atom_x``
- ``cc_contact_atom_y``
- ``cc_contact_atom_z``
- ``cc_contact_atom_u``
- ``cc_contact_atom_v``
- ``cc_contact_atom_w``
- ``cc_contact_atom_bv_x``
- ``cc_contact_atom_bv_y``
- ``cc_contact_atom_bv_z``
- ``cc_contact_atom_ref_bv_x``
- ``cc_contact_atom_ref_bv_y``
- ``cc_contact_atom_ref_bv_z``
- ``cc_contact_atom_ref_bv_r``
- ``cc_contact_atom_ref_bv_theta``
- ``cc_contact_atom_ref_bv_phi``
- ``fragment``
- ``fragment_x``
- ``fragment_y``
- ``fragment_z``
- ``fragment_u``
- ``fragment_v``
- ``fragment_w``
- ``fragment_e1_x``
- ``fragment_e1_y``
- ``fragment_e1_z``
- ``fragment_e1_u``
- ``fragment_e1_v``
- ``fragment_e1_w``
- ``fragment_w11_u``
- ``fragment_w11_v``
- ``fragment_w11_w``
- ``fragment_w12_u``
- ``fragment_w12_v``
- ``fragment_w12_w``
- ``fragment_w1_angle_1``
- ``fragment_w1_angle_2``
- ``fragment_e1_d_min``
- ``fragment_e2_x``
- ``fragment_e2_y``
- ``fragment_e2_z``
- ``fragment_e2_u``
- ``fragment_e2_v``
- ``fragment_e2_w``
- ``fragment_w21_u``
- ``fragment_w21_v``
- ``fragment_w21_w``
- ``fragment_w22_u``
- ``fragment_w22_v``
- ``fragment_w22_w``
- ``fragment_w2_angle_1``
- ``fragment_w2_angle_2``
- ``fragment_e2_d_min``
- ``fragment_e3_x``
- ``fragment_e3_y``
- ``fragment_e3_z``
- ``fragment_e3_u``
- ``fragment_e3_v``
- ``fragment_e3_w``
- ``fragment_w31_u``
- ``fragment_w31_v``
- ``fragment_w31_w``
- ``fragment_w32_u``
- ``fragment_w32_v``
- ``fragment_w32_w``
- ``fragment_w3_angle_1``
- ``fragment_w3_angle_2``
- ``fragment_e3_d_min``
- ``fragment_atom_species``
- ``fragment_atom_x``
- ``fragment_atom_y``
- ``fragment_atom_z``
- ``fragment_atom_u``
- ``fragment_atom_v``
- ``fragment_atom_w``
- ``fragment_atom_bv_x``
- ``fragment_atom_bv_y``
- ``fragment_atom_bv_z``
- ``fragment_atom_bv_u``
- ``fragment_atom_bv_v``
- ``fragment_atom_bv_w``
- ``fragment_atom_dzzp_min``
    
Details for each variable can be found in the `Data Extraction Procedure section <https://crystal-math.readthedocs.io/en/latest/procedure.html>`_. Each variable is described using a dictionary entry in the following format.

.. code-block :: json

    "variable_name": {
        "latex_name": string,
        "html_name": string,
        "family": string,
    	"level": integer,
    	"path": [list of strings],
    	"position_symmetry": [boolean,boolean,boolean,integer]
      }


Key Descriptions
^^^^^^^^^^^^^^^^

- ``variable_name``
    The name of the variable. Currently 127 variables are supported.

- ``latex_name``
    The name of the variable in LaTeX format used to render static ``*.png`` images.

- ``html_name``
    The name of the variable in html format used to render interactive ``*.html`` plots.

        
