Data Extraction Procedure
=========================
This section outlines the procedural setup required to run the Crystal Math software effectively. To extract the data it is necessary to prepare the input file ``input_data_extraction.txt`` with the user defined options to extract data. After that, the process is straightforward, as you can extract the data by simply executing the script ``csd_data_extraction.py``.

Directories Structure
---------------------
The source code can be executed by placing it in a parent directory, for instance, ``crystal_math``. Begin by creating the necessary directories within the main working directory (``crystal_math``):

.. code-block:: text

    crystal_math/
    ├── csd_db_analysis/
    │   ├── db_data/
    │   └── visualize/
    ├── source_code/
    │   └── input_files/
    └── source_data/
        └── cif_files/

- ``source_code``
	All ``*.py`` code files should be placed here.
	
- ``input_files``
	Place ``input_data_extraction.txt`` and ``input_data_analysis.txt`` here.
	
- ``source_data``
	Place the user-generated `fragment_list.json` here.
	
- ``cif_files``
	Any custom ``*.cif`` files should be placed here.

Files Description
-----------------
Each file in the Crystal Math software serves a specific function as outlined below:

- ``csd_data_extraction.py``
	Main file for the execution of the data extraction.

- ``csd_operations.py``
	Module to perform operations to identify and cluster CSD structure families and identify unique structures based on user-defined criteria.

- ``get_structures_list.py``
	Function to get the structures list for the analysis.

- ``create_reference_fragments.py``
	Function to convert user-generated fragments in the ``fragments_list.json`` to reference fragments in the space-fixed coordinate system, stored in ``reference_fragments_list.json``.

- ``get_structure_data.py``
	Function to perform the data extraction from the selected structures.

- ``structure_operations.py``
	Module to perform the necessary operations to each structure.

- ``maths.py``
	Module with the required mathematical functions.

- ``utilities.py``
	Module with several utility functions.

- ``io_operations.py``
	Module for the input/output operations.

The Data Extraction Input File
------------------------------
The first step is to modify the ``input_data_extraction.txt`` file based on the required criteria. The general format of the file and descriptions of each parameter are as follows:

Input File Format
^^^^^^^^^^^^^^^^^
The configuration should be specified in JSON format as shown below:

.. code-block:: json

    {
      "save_directory": "../csd_db_analysis/db_data/",
      "get_refcode_families": true,
      "cluster_refcode_families": true,
      "get_unique_structures": true,
      "get_structure_data": true,
      "get_structure_filter_data": true,
      "structure_list": ["csd-unique", "all"],
      "data_prefix": "homomolecular_crystals",
      "unique_structures_clustering_method": "energy",
      "target_species": ["C", "H", "N", "O", "F", "Cl", "Br", "I", "P", "S"],
      "target_space_groups": ["P1", "P-1", "P21", "C2", "Pc", "Cc", "P21/m", "C2/m", "P2/c", "P21/c", "P21/n", "C2/c", "P21212", "P212121", "Pca21", "Pna21", "Pbcn", "Pbca", "Pnma", "R-3", "I41/a"],
      "target_z_prime_values": [1, 2, 3, 4, 5],
      "molecule_weight_limit": 500.0,
      "crystal_type": ["homomolecular"],
      "molecule_formal_charges": [0],
      "structures_to_exclude": ["BALDUP","CEMVAS","DAGRIN","FADGEW","JIKXOT","LUQDAE","PEVLOR","TEVYAV","VIRLOY","ZEPDAZ04"],
      "center_molecule": true,
      "add_full_component": true,
      "proposed_vectors_n_max": 5
    }

Key Descriptions
^^^^^^^^^^^^^^^^

- ``save_directory``
	Specifies the directory where data will be saved. Using the default option is recommended.

- ``get_refcode_families``
	When set to ``true``, extracts all refcode families from the CSD, saving the output as ``csd_refcode_families.json`` within the ``db_data`` directory.

- ``cluster_refcode_families``
	When set to ``true``, clusters the structures for each refcode family. Results are saved as ``csd_refcode_families_clustered.json``.

- ``get_unique_structures``
	Retrieves unique structures for each cluster from the CSD and saves them as ``csd_refcode_families_unique_structures.json``.

- ``get_structure_data``
	Set to ``true``, performs data extraction on the selected structures.

- ``get_structure_filter_data``
	Set to ``true``, creates a file with the summarized properties for the structures that can be used to filter structures for the analysis.

- ``structure_list``
	Defines the types of structures to analyze. For the first key, the available options are 
	
	- ``"csd-all"`` for all structures
	- ``"csd-unique"`` for unique structures
	- ``"cif"`` for user-provided ``*.cif`` files. T
	
	The second key can get the value 
	
	- ``"all"`` to extract data for all structures matching the user defined criteria 
	
	or you can extract data from specific structures and/or specific compounds, by providing a list of the desired structures in the following format:
	
	- ``[["ACSALA", [0,1,11]], ["ACRDIN","all"],...]`` In each sublist, the first entry is the RefCode family name, and the second can be a list of specific entries such as ``[0,1,11]`` or it can be set to ``"all"`` to search for all the entries for the specific RefCode family. In the case we require to analyze specific entries, the indices must match what is available in the database. In the ``"ACSALA"`` example, the indices ``[0,1,11]`` are valid when combined with the ``"csd-all"`` key. When searching for unique structures however, the only valid keys are ``[24,32,35]`` corresponding to the lowest energy structures for each of the three known polymorphs.

- ``data_prefix``
	A prefix for the output files to help identify them.

- ``unique_structures_clustering_method``
	Currently only ``"energy"`` is supported, which selects structures with the lowest intermolecular lattice energy.

- ``target_species``
	List of allowed atomic species. Structures not containing these are discarded.

- ``target_space_groups``
	Specifies allowable space groups.

- ``target_z_prime_values``
	Filters structures by :math:`Z^{\prime}`.

- ``molecule_weight_limit``
	Maximum allowable molecular weight per component in the asymmetric unit.

- ``crystal_type``
	A list for the type of crystal structures to analyze. Options include ``"homomolecular"``, ``"co-crystal"``, ``"hydrate"``.

- ``molecule_formal_charges``
	Allowed molecular charges; typically set to ``[0]`` for neutral structures.

- ``structures_to_exclude``
	List of structures that cause kernel errors and are thus excluded.

- ``center_molecule``
	Whether to center the molecule in the unit cell (recommended).

- ``add_full_component``
	Analyzes complete components in the unit cell along with fragments.

- ``proposed_vectors_n_max``
	Maximum value for each component of a crystallographic vector, suggested value is ``5``.

Creating the Fragment List
--------------------------
The code includes a ``fragment_list.json`` file containing information on several fragments commonly encountered in molecular crystal structures. This file can be customized based on user needs. Each entry in the dictionary is formatted as follows:

Fragment Dictionary Format
^^^^^^^^^^^^^^^^^^^^^^^^^^
Below is an example of how a fragment, specifically ``"benzene"``, is described in the file:

.. code-block:: json

    "benzene": {
        "smarts": "c1ccccc1",
        "species": ["C", "C", "C", "C", "C", "C"],
        "coordinates": [
            [ 1.3750,  0.0000, 0.0000],
            [ 0.6875,  1.1908, 0.0000],
            [-0.6875,  1.1908, 0.0000],
            [-1.3750,  0.0000, 0.0000],
            [-0.6875, -1.1908, 0.0000],
            [ 0.6875, -1.1908, 0.0000]
        ],
        "mass": [12.0107, 12.0107, 12.0107, 12.0107, 12.0107, 12.0107],
        "atoms_to_align": "all"
    }

Key Descriptions
^^^^^^^^^^^^^^^^

- ``smarts``
	SMARTS notation representing the chemical structure of the fragment.

- ``species``
	List of atomic species corresponding to the atoms in the fragment.

- ``coordinates``
	Positions of the atoms in the fragment in any coordinate system. These will be automatically converted to space-fixed reference coordinates by the ``create_reference_fragments.py`` script.

- ``mass``
	List of atomic masses for each atom in the fragment.

- ``atoms_to_align``
	Specifies which atoms in the fragment to use for alignment. It designates specific atoms within the fragment for orientation synchronization with a corresponding fragment identified in a crystal structure. This approach is particularly useful for fragments that exhibit indistinguishable, mirror-image formations, such as oxygens in a structure like [#6]S(=O)(=O)[NH2], where traditional SMARTS representation may fall short. Accepts:

	- ``"all"``: Use all atoms for alignment.
	- List of integers: Specific atom indices to be used for alignment, essential in cases of mirror symmetries in the fragment structure.

Extracting Data
---------------
The data extraction process is initiated by executing the ``csd_data_extraction.py`` script. Depending on the parameters set (`get_refcode_families`, `cluster_refcode_families`, `get_unique_structures`), the script may first generate the respective JSON files. These operations are handled by functions within the ``csd_operations`` module. Once the initial tasks are completed, the script continues to extract data from the selected structures, which can be either CSD structures or ``*.cif`` files.

Initialization
^^^^^^^^^^^^^^
The process begins by creating a list of structures that will be analyzed. It then proceeds to loop over each structure to perform the following actions:

- **Create Objects**: Creates the CSD crystal and molecule objects.

- **Assign Properties**: Bond types, missing hydrogen atoms, and partial charges are assigned using:

  - ``molecule.assign_bond_types()``
  - ``molecule.add_hydrogens()``
  - ``molecule.assign_partial_charges()``
  
These methods are available in the CSD Python API.

- **Generate Atoms**: Generates the atoms using the ``molecule.atoms()`` method provided by the CSD Python API.

- **Extract Properties**: Crystal properties are extracted using the ``get_csd_crystal_properties(crystal)`` function in the ``csd_operations.py`` module, employing a solvent accessible surface probe with a radius of 1.2 Ångström. The upper limit for close contacts is defined as :math:`(r_{vdW_i} + r_{vdW_j} + 0.6)`. Atom and molecule properties are extracted using the ``get_csd_atom_and_molecule_properties(crystal, molecule, atoms)`` function.

- **Set Fragments**: Fragments in the structure are set using the ``get_csd_structure_fragments(input_parameters, structure, molecule)`` function. If "add_full_component" is set to False and the structure lacks the required fragments from the ``fragment_list.json``, the script skips to the next structure.

Loop Over Fragments
^^^^^^^^^^^^^^^^^^^
For each fragment in the structure, the algorithm performs extensive geometrical and topological analyses:

- **Rotate and Align Fragments**:

  - The reference fragment is rotated to align with the current fragment using the ``kabsch_rotation_matrix(A, B)`` function, which calculates the rotation matrix.
  - Normal vectors for the principal planes of inertia are identified in the crystallographic coordinate system.

- **Identify Vectors and Distances**:

  - For each normal vector :math:`(e_i)`, the algorithm finds two vectors from the set :math:`\mathbf{n}_c` that are closest to being perpendicular using ``vectors_closest_to_perpendicular(I, n_max)``.
  - The minimum distance of each principal inertia plane to selected reference points in the unit cell is calculated using ``distance_to_plane(point, plane_normal, plane_point, normal=False)``.

- **Contact Data**:

  - Detailed data for each contact includes the type (vdW or H-bond), length, line of sight verification, and vectors related to central and contact fragments in both Cartesian and spherical coordinates. Each contact can appear in the data file up to :math:`2\times N_A \times N_B` times, where the coefficient ``2`` accounts for the exchange between the central and the contact atom and :math:`N_A,\, N_B` is the number of fragments in which atoms :math:`A,\,B` appear. For example, in the ACSALA24 structure from the CSD database, a close contact forms between atoms :math:`\ce{C1}` and :math:`\ce{C2}`. Atom :math:`\ce{C1}` is common to both the benzene and carboxylic acid fragments, while atom :math:`\ce{C2}` is common to the benzene ring and the ester fragment. 

- **Hydrogen Bond Data**:

  - For each H-bond, the algorithm determines the donor and acceptor atoms, bond length, donor-acceptor distance, bond angle, and line of sight status.

Finally, all data gathered is written to output files, completing the data extraction process.

The Data Extraction Output Files
--------------------------------
Each structure's data is contained in a separate JSON file, stored in the folder ``db_data/"prefix"_structures``, where the ``"prefix"`` is set by the user in the input file. The file name for each structure is in the form ``"RefCode".json``, where the ``"RefCode"`` is identical to the CSD RefCode of the structure. The following section provide an explanation of each key-value pair in the JSON structure, by using as an expample the output file for structure ``ACSALA35`` is the CSD.

Structure File Description
^^^^^^^^^^^^^^^^^^^^^^^^^^

The JSON file is structured as follows:

.. code-block:: json

    {
        "crystal": {
            "str_id": "ACSALA35",
            "space_group": "P21/c",
            "z_crystal": 4.0,
            "z_prime": 1.0,
            "formula": "C9 H8 O4",
            "species": ["C", "H", "O"],
            "cell_lengths": [11.185, 6.5719, 11.146],
            "scaled_cell_lengths": [1.0, 0.5876, 0.9965],
            "cell_angles": [90.0, 96.01, 90.0],
            "cell_volume": 814.8025,
            "cell_density": 1.4686,
            "vdWFV": 0.253,
            "SAS": 0.0,
            "lattice_vectors": [
                [11.185, 0.0, 0.0],
                [0.0, 6.5719, 0.0],
                [-1.167, 0.0, 11.0847]
            ],
            "lattice_energy": {
                "total": -123.46,
                "electrostatic": 0.0,
                "vdW": -123.46,
                "vdW_attraction": -214.68,
                "vdW_repulsion": 91.223,
                "h-bond": 0.0,
                "h-bond_attraction": 0.0,
                "h-bond_repulsion": 0.0
            },
            "close_contacts": {
                "C4_F01.benzene_O1_F02.carboxylic_acid": {
                    "cc_length": 3.5464,
                    "cc_type": "vdW",
                    "cc_is_in_los": true,
                    "cc_central_atom": {
                        "atom": "C",
                        "fragment": "benzene",                        
                        "coordinates": {
                            "cartesian": [-1.6689,4.8803,-2.1349],
                            "fractional": [-0.1693,0.7426,-0.1926]
                        },
                        "bond_vectors": [-3.8744,2.4323,-3.2435],
                        "reference_bond_vectors": [0.1525,4.5461,3.28]                       
                    },
                    "cc_contact_atom": {
                        "atom": "O",
                        "fragment": "carboxylic_acid",
                        "coordinates": {
                            "cartesian": [1.4354,5.642,-0.5986],
                            "fractional": [0.1227,0.8585,-0.054]
                        },
                        "bond_vectors": [-0.7701,3.194,-1.7072],
                        "reference_bond_vectors": [-1.0013,3.5639,0.0735],
                        "reference_bond_vectors_spherical": [3.7027,88.8629,105.6929]            
                    }
                },
                // ...
            }
            "hbonds": {
                "O1_H1_O2": {
                    "hb_atoms": ["O","H","O"],
                    "hb_length": 1.6839,
                    "hb_da_distance": 2.6421,
                    "hb_angle": 159.0931,
                    "hb_is_in_los": true,
                    "hb_donor_coordinates": [1.4354,5.642,-0.5986],
                    "hb_h_coordinates": [1.0214,6.552,-0.6131],
                    "hb_acceptor_coordinates": [-0.0122,7.8028,-1.063]
                }
            }
        },
        "fragments": {
            "F01.benzene": {
                "fragment": "benzene",
                "coordinates": {
                    "cartesian": [2.2055,2.448,1.1086],
                    "fractional": [0.2076,0.3725,0.1]
                },                
                "inertia_planes": {
                    "e_1": {
                        "cartesian": [-0.6975,-0.1026,0.7092],
                        "crystallographic": [-0.6676,-0.0577,0.7423],
                        "perpendicular_vectors": {
                            "vector_1": [1,0,1],
                            "vector_2": [5,0,4],
                            "angle_1": 93.03,
                            "angle_2": 86.7
                        },
                        "min_distance_to reference_points": 0.0081
                    },
                    // ...
                },
                "atoms": {
                    "C2": {
                        "species": "C",
                        "coordinates": {
                            "cartesian": [1.6445,3.6934,0.7305],
                            "fractional": [0.1539,0.562,0.0659]
                        },
                        "bond_vectors": {
                            "cartesian": [-0.561,1.2454,-0.3781],
                            "fractional": [-0.0537,0.1895,-0.0341]
                        },
                        "dzzp_min": 0.0028
                    },
                    // ...
                }
            }
        }
    }

Key descriptions
^^^^^^^^^^^^^^^^

- ``crystal``
	Contains all data specific to the crystal structure.

- ``str_id``
	A unique identifier for the structure.

- ``space_group``
	The space group of the crystal structure.

- ``z_crystal``
	The number of formula units per unit cell.

- ``z_prime``
	The number of asymmetric units in the crystal structure.

- ``formula``
	The chemical formula of the crystal.

- ``species``
	A list of unique atomic species present in the crystal.

- ``cell_lengths``
	The lengths of the cell edges :math:`(a, b, c)`.

- ``scaled_cell_lengths``
	Cell lengths scaled relative to the longest cell edge.

- ``cell_angles``
	The angles between the cell edges :math:`(\alpha, \beta, \gamma)`.

- ``cell_volume``
	The volume of the crystal's unit cell.

- ``cell_density``
	The density of the crystal calculated from the unit cell volume and formula weight.

- ``vdWFV``
	Van der Waals fraction volume.

- ``SAS``
	Surface area to volume ratio.

- ``lattice_vectors``
	A list of the three lattice vectors defining the unit cell.

- ``lattice_energy``
	Contains various components of the calculated lattice energy.
	
	- ``total``: The total lattice energy.
	- ``electrostatic``: The electrostatic contribution to the lattice energy.
	- ``vdW``: The vdW contribution to the lattice energy.
	- ``vdW_attraction``: The attractive vdW contribution to the lattice energy.
	- ``vdW_repulsion``: The respulsive vdW contribution to the lattice energy.
	- ``h-bond``: The hbond contribution to the lattice energy.
	- ``h-bond_attraction``: The attractive hbond contribution to the  lattice energy.
	- ``h-bond_repulsion``: The repulsive hbond contribution to the lattice energy.
	
- ``close_contacts``
	Details of close atomic contacts within the crystal structure.
	
	- ``XA_FA_YB_FB``: The label for the contact (labels of the atoms and the respective fragments in the structure).
	
		- ``cc_length``: The length of the contact in Angstroms.
		- ``cc_type``: The type of the contact (``vdW`` or ``hbond``).
		- ``cc_is_in_los``: If the contact is in line of sight (``true`` of ``false``).
		- ``cc_central_atom``: The details for the central atom of the contact pair.
		
			- ``atom``: The species of the central atom.
			- ``fragment``: The fragment of the central atom.
			- ``coordinates``: The coordinates of the central atom (``cartesian`` and ``fractional``).
			- ``bond_vetors``: The cartesian bond vectors for the central atom relative to the center of mass of the fragment.
			- ``reference_bond_vetors``: The cartesian bond vectors for the central atom relative to the center of mass of the fragment in the inertia frame of the fragment.
			
		- ``cc_contact_atom``: The details for the contact atom of the contact pair.
		
			- ``atom``: The species of the central atom.
			- ``fragment``: The fragment of the central atom.
			- ``coordinates``: The coordinates of the central atom (``cartesian`` and ``fractional``).
			- ``bond_vetors``: The cartesian bond vectors for the central atom relative to the center of mass of the fragment.
			- ``reference_bond_vetors``: The cartesian bond vectors for the central atom relative to the center of mass of the fragment in the inertia frame of the fragment.
			- ``reference_bond_vetors_spherical``: The bond vectors in spherical coordinates for the central atom relative to the center of mass of the fragment in the inertia frame of the fragment.

- ``hbonds``
	Details of hydrogen bonds within the crystal structure.
	
	- ``XA_HB_YC``: The hbond label.
	
		- ``hb_atoms``: A list of the atomic species forming the hydrogen bond. The first atom coorespond to the donor and the thord to the acceptor of the bond.
		- ``hb_length``: The length of the hydrogen bond in Angstroms.
		- ``hb_da_distance``: The donor-acceptor distance in Angstroms.
		- ``hb_angle``: The angle of the hydrogen bond.
		- ``hb_is_in_los``: : If the hydrogen bond is in line of sight (``true`` of ``false``).
		- ``hb_donor_coordinates``: The cartesian coordinates of the donor atom.
		- ``hb_h_coordinates``: The cartesian coordinates of the hydrogen atom.
		- ``hb_acceptor_coordinates``: The cartesian coordinates of the acceptor atom.

- ``fragments``
	Details of individual molecular or ionic fragments within the structure, including coordinates and properties.
	
	- ``FXX.fragment_name``: The label for the fragment.
	
		- ``fragment``: The fragment name.
		- ``coordinates``: The coordinates for the center of mass of the fragment (``cartesian`` and ``fractional``).
		- ``inertia_planes``: The details for the inertia planes of the fragments.
		
			- ``e_i``: The label of the inertia plane (:math:`i=1,2,3`).
				
				- ``cartesian``: The normal vector in the cartesian coordinate system.
				- ``crystallographic``: The normal vector in the crystallographic coordinate system.
				- ``perpendicular_vectors``: Details for the near-perpendicular vectors from the set :math:`\mathbf{n}_c`.
					
					- ``vector_1``, ``vector_2``: The components of the two near-perpendicular vectors from the set :math:`\mathbf{n}_c`.
					- ``angle_1``, ``angle_2``: The angles between the vector ``e_i`` and ``vector_1``, ``vector_2`` respectively.
					
				- ``min_distance_to_reference_points``: The minimum distance of the inertia plane to the reference points of the unit cell.
				
		- ``atoms``: The details for the atoms comprising the fragment.
			
			- ``XA``: The label of the atom.
			
				- ``species``: The species of the atom.
				- ``coordinates``: The coordinates for the atom (``cartesian`` and ``fractional``).
				- ``bond_vectors``: The bond vectors of the atom to the center of mass of the fragment (``cartesian`` and ``fractional``).
				- ``dzzp_min``: The minimum distance of the atom to the ZZP plane family.

Data Filtering File Description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The algorithm also generates a file ``"prefix"_structures_filter_data.json`` within the ``db_data`` folder, that contains compact information for each structures that can be used to rapidly filter structures in the post extraction analysis step. Each structure is represented as a dictionary entry, with the key being identical to the CSD RefCode of the structure. The format for each entry is as follows.

.. code-block:: json 

    "ACSALA35": {
        "space_group": "P21/c",
        "z_crystal": 4.0,
        "z_prime": 1.0,
        "species": ["C","H","O"],
        "fragments": ["benzene","carboxylic_acid","ester_aromatic-aliphatic"],
        "contact_pairs": [
            ["C","O","vdW",false],
            // ...
        ],
        "contact_central_fragments": [
            ["benzene","vdW",false],
            // ...
        ],
        "contact_fragment_pairs": [
            ["benzene","carboxylic_acid","vdW",false],
            // ...
        ]
    }

Key descriptions
^^^^^^^^^^^^^^^^

- ``space_group``
	The space group of the structure 
	
- ``z_crystal``
	The total number of molecules :math:`Z` in the reference unit cell. For :math:`Z^{\prime}=1` this number is identical to the symmetry operations of the space group. 
	
- ``z_prime``
	The numner :math:`Z^{\prime}` of molecules in the asymmetric unit
	
- ``species``
	A list of the different atomic species found in the structure.
	
- ``fragments``
	A list of the different fragments found in the structure.
	
- ``contact_pairs``
	A list of the different close contact atomic pairs found in the structure. The first entry is the central atom, the second the contact atom, the third entry the type of the contact and the fourth declares if the contact is in line of sight.
	
	
- ``contact_central_fragments``
	A list of the different central fragments for the close contacts in the structure. The first entry is the central fragment, the second entry the type of the contact and the third declares if the contact is in line of sight.	
	
- ``contact_fragment_pairs``
	A list of the different close contact fragment pairs found in the structure. The first entry is the central fragment, the second the contact fragment, the third entry the type of the contact and the fourth declares if the contact is in line of sight.
	
Example Usage
-------------
In the following paragraphs we demostrate the workflow for extracting sample data from the CSD. We show how to extract data for all the unique aspirin structures as well as for two known :math:`Z^{\prime}=1` acridine polymorphs. We will perform the extraction in two different steps: 

- **General CSD structure identification**: This part of the extraction will generate the files ``csd_refcode_families.json``, ``csd_refcode_families_clustered.json``, ``csd_refcode_families_unique_structures.json`` for the user-defined settings in the input file. The input file for this operation will be

	.. code-block:: json

		{
		  "save_directory": "../csd_db_analysis/db_data/",
		  "get_refcode_families": true,
		  "cluster_refcode_families": true,
		  "get_unique_structures": true,
		  "get_structure_data": false,
		  "get_structure_filter_data": false,
		  "structure_list": ["csd-unique", "all"],
		  "data_prefix": "example",
		  "unique_structures_clustering_method": "energy",
		  "target_species": ["C", "H", "N", "O", "F", "Cl", "Br", "I", "P", "S"],
		  "target_space_groups": ["P1", "P-1", "P21", "C2", "Pc", "Cc", "P21/m", "C2/m", "P2/c", "P21/c", "P21/n", "C2/c", "P21212", "P212121", "Pca21", "Pna21", "Pbcn", "Pbca", "Pnma", "R-3", "I41/a"],
		  "target_z_prime_values": [1, 2, 3, 4, 5],
		  "molecule_weight_limit": 500.0,
		  "crystal_type": ["homomolecular"],
		  "molecule_formal_charges": [0],
		  "structures_to_exclude": ["BALDUP","CEMVAS","DAGRIN","FADGEW","JIKXOT","LUQDAE","PEVLOR","TEVYAV","VIRLOY","ZEPDAZ04"],
		  "center_molecule": true,
		  "add_full_component": true,
		  "proposed_vectors_n_max": 5
		}
		
	Note that ``get_structure_data`` and ``get_structure_filter_data`` are set to ``false``. This process will create a list of structures that are consistent with the filters

	- ``target_species``,
	- ``target_space_groups``,
	- ``target_z_prime_values``,
	- ``molecular_weight_limit``,
	- ``crystal_type``,
	- ``molecule_formal_charges``.

	This set of structures is recommended to be as general as possible, so that in can be used for data extraction without having to identify and cluster structures every time we perform a data extraction. Thus, while in the example we are dealing with :math:`Z^{\prime}=1,\,2` structures comprising of C, H, N, O atoms in the :math:`P2_1/c,\,P2_1/n` space groups, we keep the respective filters more general to include a high number of structures of interest for subsequent analysis. This step must be exectuted only in case we need to include more structures in the files ``csd_refcode_families.json``, ``csd_refcode_families_clustered.json``, ``csd_refcode_families_unique_structures.json``, for example when we need to expand the filters or when a CSD update is released. With the default options for the filters, this process generates a list of ~230,000 unique structures that are sufficient for subsequent statistical analysis. 
	
- **Data extraction for structures of interest**

	In the above input file, setting ``get_structure_data = true`` will extract data for all the unique structures identified in the previous step. In this example however, we want to extract data for a small subset of structures: the three known aspirin polymorphs and the two known :math:`Z^{\prime}=1` acridine polymorphs. By checking the file ``csd_refcode_families_unique_structures``, we can see three entries for aspirin (``ACSALA24``, ``ACSALA32`` and ``ACSALA35``) and 6 entries for acridine, with ``ACDRIN11`` and ``ACRDIN12`` being the :math:`Z^{\prime}=1` polymorphs. We modify the input file to extract data for this small set of structures.
	
	.. code-block:: json

		{
		  "save_directory": "../csd_db_analysis/db_data/",
		  "get_refcode_families": false,
		  "cluster_refcode_families": false,
		  "get_unique_structures": false ,
		  "get_structure_data": true,
		  "get_structure_filter_data": true,
		  "structure_list": ["csd-unique", [["ACSALA", "all"], ["ACRDIN", [11,12]]]],
		  "data_prefix": "example",
		  "unique_structures_clustering_method": "energy",
		  "target_species": ["C", "H", "N", "O", "F", "Cl", "Br", "I", "P", "S"],
		  "target_space_groups": ["P1", "P-1", "P21", "C2", "Pc", "Cc", "P21/m", "C2/m", "P2/c", "P21/c", "P21/n", "C2/c", "P21212", "P212121", "Pca21", "Pna21", "Pbcn", "Pbca", "Pnma", "R-3", "I41/a"],
		  "target_z_prime_values": [1, 2, 3, 4, 5],
		  "molecule_weight_limit": 500.0,
		  "crystal_type": ["homomolecular"],
		  "molecule_formal_charges": [0],
		  "structures_to_exclude": ["BALDUP","CEMVAS","DAGRIN","FADGEW","JIKXOT","LUQDAE","PEVLOR","TEVYAV","VIRLOY","ZEPDAZ04"],
		  "center_molecule": true,
		  "add_full_component": true,
		  "proposed_vectors_n_max": 5
		}
		
	Note that ``get_refcode_families``, ``cluster_refcode_families``, ``get_unique_structures`` are all set to ``false``. In the ``structure_list`` key, for the aspirin structures we select to extract data for all unique structures (``["ACSALA", "all"]``), while for the acridine we select to extract data only for entries 11 and 12 (``["ACRDIN", [11,12]]``).
	
	The algorithm will generate the output files for the 5 structures as well as the file ``example_structures_filter_data.json`` with the compact structure information. For your convenience, these structures are provided in the `project's GitHub page <https://github.com/nigalanakis/Crystal_Math/tree/master/docs/examples>`_.