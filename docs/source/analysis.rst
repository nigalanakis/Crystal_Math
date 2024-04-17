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
      "plots_directory": "../csd_db_analysis/vizualize/",
      "data_directory": "../csd_db_analysis/db_data/",
      "data_prefix": "homomolecular",
      "figure_size": [5, 3.75],
      "plot_data_options": {
        "space_groups": "all",
        "atom_types": "all",
        "individual_space_groups_plots": "true",
        "fragments": "all",
        "contact_fragment_pairs": "all",
        "contact_atom_pairs": "all",
        "contact_type": "vdW",
        "scatter": [["a", "b"]],
        "scatter_marker": "o",
        "scatter_facecolor": "whitesmoke",
        "scatter_edgecolor": "black",
        "scatter_alpha": 1.0,
        "scatter_lims": [["custom", "custom"],["custom", "custom"]],
        "histogram": [["a", "false"],["b", "true"]],
        "histogram_lims": ["custom", "custom"],
        "titles": "true"
      },
      "save_figs": "false"
    }

Key Descriptions
^^^^^^^^^^^^^^^^
- ``"plots_directory"``: Specifies the directory where plots will be saved. Using the default option is recommended.
- ``"data_directory"``: The directory where the extracted data is stored. It must match the ``"save_directory"`` specified in the ``input_data_extraction.json`` file.
- ``"data_prefix"``: A prefix applied to output files to facilitate their identification. This must be consistent with the ``"data_prefix"`` in the ``input_data_extraction.json`` file.
- ``"figure_size"``: Defines the dimensions of exported figures in inches, formatted as :math:`(W \times H)`. The default Matplotlib size is :math:`(6.4 \times 4.8)`. To place two figures side by side in a 12-inch wide document using an 11pt font, the optimal size is :math:`(5.0 \times 3.75)`. Adjust dimensions according to your document's specific requirements.
- ``"plot_data_options"``: A collection of settings for generating scatter plots and histograms:

  - ``"space_groups"``: Determines which space groups to include in the plots. Options include:
    
    - ``"all"``: Analyses data for all structures in the data files.
    - A specific list of space groups, e.g., ``["P21/c", "P21/n"]``: Includes data only for structures in the selected space groups.

  - ``"atom_types"``: Specifies which atomic species will be included in the data. Options include:
    
    - ``"all"``: Analyses data for all structures in the data files.
    - A specific list of atomic species, e.g., ``["C", "N" "O"]``: Analyses data only for structures including at least one of the selected atomic species.

  - ``"individual_space_groups_plots"``: Indicates whether plots should be generated for each space group separately.
  
  - ``"fragments"``: Specifies the fragments to be included in the data. Options include:
    
    - ``"all"``:Analyses data for all structures in the data files.
    - A specific list of fragments, e.g., ``["benzene", "carboxylic_acid"]``: Analyses data for all structures in the data files including at least one of the selected fragments. 

  - ``"contact_fragment_pairs"``: Determines which pairs of contact fragments to include in the data Options include:
    
    - ``"all"``: Includes all contact fragment pairs from the contact data files.
    - A specific list of fragment pairs, e.g., ``[["benzene", "benzene"], ["benzene", "carboxylic_acid"]]``: Analyses data for all structures in the data files including at least one of the selected fragment pairs. The first fragment in each pair is the central fragment, and the second is the contact fragment.

  - ``"contact_atom_pairs"``: Specifies which atomic pairs to include in the  data. Options include:
 
    - ``"all"``: Includes all contact pairs found in the contact data files.
    - A specific list of atomic pairs, e.g., ``[["C", "H"], ["H", "O"]]``: Analyses data for all structures in the data files including at least one of the selected contact atom pairs. The first atom in each pair is associated with the central fragment, and the second with the contact fragment.
	
  - ``"contact_type"``: The type of the contacts in the contact data file that will be considered in the analysis. Options include:
  
    - ``"all"``: Both vdW and H-bond contacts will be considered in the analysis.
    - ``"vdW"``: Only vdW contacts will be considered in the analysis.
    - ``"hbond"``: Only H-bond contacts will be considered in the analysis.

  - ``"scatter"``: A list of variable pairs to generate scatter plots, eg ``[["a", "b"], ["hb_length", "DA_dis"]]``. For a full list of the available variables please refer to the following section.  
  - ``"scatter_marker"``: The marker shape for the scatter plot. Any marker available in the Matplotlib package can be used. For a list of the available markers, please refer to the `official Matplotlib markers guide <https://matplotlib.org/stable/api/markers_api.html>`_.
  - ``"scatter_facecolor"`` and ``"scatter_edgecolor"``: The marker face and edge colors for the scatter plot. Any named color available in the Matplotlib package can be used. For a list of the available colors, please refer to the `official Matplotlib color guide <https://matplotlib.org/stable/gallery/color/named_colors.html>`_.
  - ``"scatter_alpha"``: The opacity of the data points in the scatter plot. Can take any value between ``0.0`` (100% opacity) to ``1.0`` (0% opacity).
  - ``"scatter_lims"``: The limits of the axes for the scatter plot in the format ``[[x_min, x_max],[y_min, y_max]]``. Any value can be set to: 
    
    - ``"auto"``: The limits for the axis will be determined automaticaly by Matplotlib package.
    - ``"custom"``: The limits will be determined manually using a 10% buffer to the minimum and maximum values of the variables in the full dataset. For a specific variable pair, eg ``["a", "b"]``, the limits of the plots will be identical for all space groups. This option is useful when comparing results between different space groups.
	
  - **ADD HISTOGRAM OPTIONS HERE**
  - ``"titles"``: Declares whwther a title will be added in the plots or not.
  
- ``"save_figs"``: Declares if the generated plots will be saved. If set to ``"true"``, the plots are saved in ``*.png`` format with a 300dpi resolution. The name for each plot is characteristic of the variables included in each plot.
  
  - **scatter plots** are saved in the format: ``"data_prefix" + _scatter_plot_ + "variable_1" + _vs_ + "variable_2" + _ + "space_group" + .png``, for example ``homomolecular_scatter_plot_a_vs_b_P21c.png``.    
  - **histograms** are saved in the format: ``"data_prefix" + _histogram_ + "variable" + _ + "space_group" + .png``, for example ``homomolecular_histogram_a_P21c.png``.     
  
Applying filters
^^^^^^^^^^^^^^^^
The algorithm offers a series of filters to work with:

- The complete set of data for all structures, fragments, contacts, and hydrogen bonds,
- Structures in a user-defined subset of space groups,
- Structures including a user-defined subset of fragments,
- Structures including a user-defined subset of central/contact fragment pairs,
- Structures including a user-defined subset of contacts based on the species of the atoms in the central and contact fragments,
- Structures including specific types of close contacts (vader Waals, hydrogen bonds) **ADD HALOGEN BONDS**. 
- Structures including specific types of atomic species. **TO BE ADDED**

By combining the available filters, it is possible to perform analyis in a refined subset of data based on the needs of the project. 

- **EXAMPLES TO BE ADDED HERE**

List of Variables
-----------------
Below is a list of all variables available in the data files. Please note that for scatter plots, only pairs of variables with the same size can be used, so it is not possible to combine variables from different files.

- **Structure data variables**

  - ``"a"``, ``"b"``, ``"c"``: The cell lengths :math:`a,\, b,\, c` in Angstrom. 
  - ``"a_sc"``, ``"b_sc"``, ``"c_sc"``: The dimensionless scaled cell lengths :math:`a_s=1.0,\, b_s=b/a,\, c_s=c/a`.
  - ``"alpha"``, ``"beta"``, ``"gamma"``: The cell angle :math:`\alpha,\,\beta,\,\gamma` in degrees.
  - ``"volume"``: The unit cell volume :math:`\Omega` in Angstrom\ :math:`^3`.
  - ``"density"``: The unit cell density :math:`\rho` in gr/cm\ :math:`^3`. 
  - ``"vdWFV"``, ``"SAS"``: The vdW free volume and the solvent accessible surface of the structure as a percentage of the total unit cell volume.
  - ``"E_tot"``: The total lattice energy of the crystal in kJ/mol.
  - ``"E_el"``: The total electrostatic energy of the crystal in kJ/mol.
  - ``"E_vdW"``, ``"E_vdW_at"``, ``"E_vdW_rep"``: The total, the attractive and repulsive component of the vdW energy of the crystal in kJ/mol.
  - ``"E_hb"``, ``"E_hb_at"``, ``"E_hb_rep"``:	The total, the attractive and the repulsive component of the hydrogen bond energy of the crystal in kJ/mol.
  
- **Fragment data variables** 

  - ``"x"``, ``"y"``, ``"z"``: The cartesian coordinates for the center of mass of the fragment (hydrogen atoms aer excluded).  
  - ``"u"``, ``"v"``, ``"w"``: The fractional coordinates for the center of mass of the fragment (hydrogen atoms aer excluded).
  - ``"ei_x"``, ``"ei_y"``, ``"ei_z"``, :math:`i=1,2,3`: The cartesian components of the principal axes of inertia for the fragment.
  - ``"ei_u"``, ``"ei_v"``, ``"ei_w"``, :math:`i=1,2,3`: The normal vector components of the crystallographic principal planes of inertia for the fragment.
  - ``"di"``, :math:`i=1,2,3`: The minimum distance of the principal planes of inertia to reference cell points
  - ``"Wij_u"``, ``"Wij_v"``, ``"Wij_w"``, :math:`i=1,2,3`, :math:`j=1,2`: The components of the crystallographic vectors from the set :math:`\mathbf{n}_c` that are closest to be perpendicular to the crystallographic vectors :math:`e_i`. 
  - ``"ang_ij"``, :math:`i=1,2,3`, :math:`j=1,2`: The angles between the vectors :math:`W_{ij}` to the normal vectors of the crystallographic principal axes of inertia.
  - ``"at_x"``, ``"at_y"``, ``"at_z"``: The cartesian coordinates of the atoms comprising the fragment.
  - ``"at_u"``, ``"at_v"``, ``"at_w"``: The fractional coordinates of the atoms comprising the fragment. 
  - ``"dzzp_min"``: The minimum distance of the atom to the nearest plane of the ZZP planes family.
  
- **Contact data variables** 

  - ``"cc_length"``: The length of the close contact in Angstroms. 
  - ``"speci"``, :math:`i=1,2`: The species of the 2 atoms forming the contact. :math:`i=1` refers to the atom in the central fragment and :math:`i=2` to the atom in the contact fragment. 
  - ``"xi"``, ``"yi"``, ``"zi"``, :math:`i=1,2`: The cartesian coordinates of the 2 atoms forming the contact. :math:`i=1` refers to the atom in the central fragment and :math:`i=2` to the atom in the contact fragment.  
  - ``"bvxi"``, ``"bvyi"``, ``"bvzi"``, :math:`i=1,2`: The cartesian coordinates of the of the 2 atoms forming the contact, relative to the center of mass of the central fragment. :math:`i=1` refers to the atom in the central fragment and :math:`i=2` to the atom in the contact fragment.  
  - ``"bvxi_ref"``, ``"bvyi_ref"``, ``"bvzi_ref"``, :math:`i=1,2`: The cartesian coordinates of the of the 2 atoms forming the contact, relative to the center of mass of the central fragment in the inertia frame. :math:`i=1` refers to the atom in the central fragment and :math:`i=2` to the atom in the contact fragment.   
  - ``"r2"``, ``"theta2"``, ``"phi2"``, :math:`i=1,2`: The spherical coordinates of the atom in the contact fragment, relative to the center of mass of the central fragment in the inertia frame.   
  
- **Hydrogen bond data variables** 

  - ``"hb_length"``: The length of the close contact in Angstroms. 
  - ``"DA_dis"``: The donor-acceptor of the close contact in Angstroms. 
  - ``"angle"``: The angle of the hydrogen bond in angstroms. 
  - ``"specD"``, ``"specA"``: The species of the donor and the acceptor atoms forming the hydrogen bond.
  - ``"xD"``, ``"yD"``, ``"zD"``: The cartesian coordinates for the donor atom. 
  - ``"xH"``, ``"yH"``, ``"zH"``: The cartesian coordinates for the hydrogen atom.
  - ``"xA"``, ``"yA"``, ``"zA"``: The cartesian coordinates for the acceptor atom.
