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
     "data_prefix": "01_test",
     "figure_size": [5,3.75],
     "data_filters": {
         "space_group": {
             "is_active": true,
             "type": "single",
             "values": ["P21/c"],
             "operator": "or",
             "refine_data": false
         }, 
         "z_crystal": {
             "is_active": false,
             "type": "single",
             "values": [4],
             "operator": "or",
             "refine_data": false
         }, 
         "z_prime": {
             "is_active": true,
             "type": "single",
             "values": [1,2],
             "operator": "or",
             "refine_data": false
         },
         "species": {
             "is_active": true,
             "type": "multiple",
             "values": ["C","H","O"],
             "operator": "or",
             "refine_data": false
         },
         "fragments": {
             "is_active": true,
             "type": "multiple",
             "values": [
                 "benzene",
                 "carboxylic_acid"
                 ],
             "operator": "or",
             "refine_data": true
         },
         "contact_pairs": {
             "is_active": true,
             "type": "multiple_lists",
             "values": [
                 ["H","O","hbond",true]
                 ],
             "operator": "or",
             "refine_data": true
         },
         "contact_central_fragments": {
             "is_active": false,
             "type": "multiple_lists",
             "values": [
                 ["carboxylic_acid","vdW",true],
                 ["ester_aromatic-aliphatic","vdW",true]
                 ],
             "operator": "and",
             "refine_data": true
             },
         "contact_fragment_pairs": {
             "is_active": true,
             "type": "multiple_lists",
             "values": [
                ["carboxylic_acid","carboxylic_acid","hbond",false],
                ["carboxylic_acid","carboxylic_acid","hbond",true]
                 ],
             "operator": "and",
             "refine_data": true
             }
     },
     "plot_data_options": {
        "individual_space_groups_plots": true,
        "scatter": [
            ["cell_length_a", "cell_length_b"], 
            ["cell_length_b_sc", "cell_length_c_sc"], 
            ["vdWFV", "E_tot"]
            ],
        "scatter_marker": "o",
        "scatter_facecolor": "whitesmoke",
        "scatter_edgecolor": "black",
        "scatter_opacity": 1.0,
        "scatter_lims": [
            ["custom", "custom"], 
            ["custom", "custom"]
            ],
        "3D_scatter": [
            ["cell_length_a", "cell_length_b", "cell_length_c"]
            ],
        "3D_scatter_marker": "o",
        "3D_scatter_facecolor": "whitesmoke",
        "3D_scatter_edgecolor": "black",
        "3D_scatter_opacity": 1.0,
        "3D_scatter_lims": [
            ["custom", "custom"], 
            ["custom", "custom"], 
            ["custom", "custom"]
            ],
        "histogram": [
            ["fragment_x", false],
            ["fragment_atom_x", false], 
            ["cc_length", false]
            ],
        "histogram_lims": ["custom", "custom"],
        "titles": false
     },
     "save_figs": false
    }


Key Descriptions
^^^^^^^^^^^^^^^^
- ``plots_directory``: Specifies the directory where plots will be saved. Using the default option is recommended.
- ``data_directory``: The directory where the extracted data is stored. It must match the ``"save_directory"`` specified in the ``input_data_extraction.json`` file.
- ``data_prefix``: A prefix applied to output files to facilitate their identification. This must be consistent with the ``"data_prefix"`` in the ``input_data_extraction.json`` file.
- ``figure_size``: Defines the dimensions of exported figures in inches, formatted as :math:`(W \times H)`. The default Matplotlib size is :math:`(6.4 \times 4.8)`. To place two figures side by side in a 12-inch wide document using an 11pt font, the optimal size is :math:`(5.0 \times 3.75)`. Adjust dimensions according to your document's specific requirements.
- ``data_filters``: Details for filtering structures for the analysis. Structures can be filtered based on 

	- **Space group**: The space group of the structure.
	- :math:`Z` **value**. The total number of molecules in the unit cell (Number of symmetry operations) :math:`\times` (Number of molecules in the asymmetric unit).
	- :math:`Z^{\prime}` **value**: The number of molecules in the asymmetric unit.
	- **Atomic species**: The different atomic species found in the structure.
	- **Fragments**: The different fragments found in the structure.
	- **Contact atomic pairs**: The different atomic pairs found for the contacts in the structure.
	- **Contact central fragments**: The different central fragments for the contacts in the structure.
	- **Contact fragment pairs**: The different fragment pairs found for the contacts in the structure.
	
	Each filter has 5 options:

	- ``is_active``: Set to ``true`` to activate the filter. Setting to ``false`` will deactivate the filter.
	- ``type``: The type of the filter. The available options are 
	
		- ``single``: A structure is characterized by a single specific value for the variable (for example the space group).
		- ``multiple``: A structure is characterized by a list of values for the specific variable (for example the atomic species in the structure).
		- ``multiple_list``: A structure is characterized by a list of values for the specific variable, but each value is now a list (for example the contact pairs in the structure, where each contact pair is characterized by the species of the cetnral atom, the species of the contact atom, the type of the contact and a boolean that states if the contact is in line of sight).
	
	- ``values``: A list (or a list of lists) for the allowed values.
	- ``operator``: The available options are
	
		- ``"or"``: The filter will check for structures that have **any** of the declared values,
		- ``"and"``: The filter will check for structures that have **all** the declared values,
		
    - ``refine_data``: Set to ``true`` to refine the data for all the components in the structure based on the values of the filter. 
         