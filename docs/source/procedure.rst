Procedure
=========
This section outlines the procedural setup required to run the Crystal Math software effectively.

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

- ``source_code``: All ``*.py`` code files should be placed here.
- ``input_files``: Place ``input_data_extraction.txt`` and ``input_data_analysis.txt`` here.
- ``source_data``: Place the user-generated `fragment_list.json` here.
- ``cif_files``: Any custom ``*.cif`` files should be placed here.

Files Description
-----------------
Each file in the Crystal Math software serves a specific function as outlined below:

- ``csd_data_extraction.py``: Main file for the execution of the data extraction.
- ``csd_operations.py``: Module to perform operations to identify and cluster CSD structure families and identify unique structures based on user-defined criteria.
- ``get_structures_list.py``: Function to get the structures list for the analysis.
- ``create_reference_fragments.py``: Function to convert user-generated fragments in the ``fragments_list.json`` to reference fragments in the space-fixed coordinate system, stored in ``reference_fragments_list.json``.
- ``get_structure_data.py``: Function to perform the data extraction from the selected structures.
- ``structure_operations.py``: Module to perform the necessary operations to each structure.
- ``maths.py``: Module with the required mathematical functions.
- ``utilities.py``: Module with several utility functions.
- ``io_operations.py``: Module for the input/output operations.

The Data Extraction Input File
------------------------------
The first step is to modify the ``input_data_extraction.txt`` file based on the required criteria. The general format of the file and descriptions of each parameter are as follows:

Input File Format
^^^^^^^^^^^^^^^^^
The configuration should be specified in JSON format as shown below:

.. code-block:: json

    {
      "save_directory": "../csd_db_analysis/db_data/",
      "get_refcode_families": "true",
      "cluster_refcode_families": "true",
      "get_unique_structures": "true",
      "get_structure_data": "true",
      "structure_list": ["csd-unique", "all"],
      "data_prefix": "homomolecular_crystals",
      "unique_structures_clustering_method": "energy",
      "target_species": ["C", "H", "N", "O", "F", "Cl", "Br", "I", "P", "S"],
      "target_space_groups": ["P1", "P-1", "P21", "C2", "Pc", "Cc", "P21/m", "C2/m", "P2/c", "P21/c", "P21/n", "C2/c", "P21212", "P212121", "Pca21", "Pna21", "Pbcn", "Pbca", "Pnma", "R-3", "I41/a"],
      "target_z_prime_values": [1, 2, 3, 4, 5],
      "molecule_weight_limit": 500.0,
      "crystal_type": "homomolecular_crystal",
      "molecule_formal_charges": [0],
      "structures_to_exclude": ["BALDUP","CEMVAS","DAGRIN","FADGEW","JIKXOT","LUQDAE","PEVLOR","TEVYAV","VIRLOY","ZEPDAZ04"],
      "center_molecule": "true",
      "add_full_component": "true",
      "proposed_vectors_n_max": 5
    }

Key Descriptions
^^^^^^^^^^^^^^^^
- ``"save_directory"``: Specifies the directory where data will be saved. Using the default option is recommended.
- ``"get_refcode_families"``: When set to ``"true"``, extracts all refcode families from the CSD, saving the output as ``csd_refcode_families.json`` within the ``db_data`` directory.
- ``"cluster_refcode_families"``: When set to ``"true"``, clusters the structures for each refcode family. Results are saved as ``csd_refcode_families_clustered.json``.
- ``"get_unique_structures:``: Retrieves unique structures for each cluster from the CSD and saves them as ``csd_refcode_families_unique_structures.json``.
- ``"get_structure_data:``: set to ``"true"``, performs data extraction on the selected structures.
- ``"structure_list"``: Defines the types of structures to analyze. Can specify ``"csd-all"`` for all structures, ``"csd-unique"`` for unique structures, or ``"cif"`` for user-provided ``*.cif`` files.
- ``"data_prefix:``: A prefix for the output files to help identify them.
- ``"unique_structures_clustering_method:``: Currently only ``"energy"`` is supported, which selects structures with the lowest intermolecular lattice energy.
- ``"target_species"``: List of allowed atomic species. Structures not containing these are discarded.
- ``"target_space_groups"``: Specifies allowable space groups.
- ``"target_z_prime_values"``: Filters structures by :math:`Z^{\prime}`.
- ``"molecule_weight_limit"``: Maximum allowable molecular weight per component in the asymmetric unit.
- ``"crystal_type"``: Type of crystal structures to analyze. Options include ``"homomolecular"``, ``"co-crystal"``, ``"hydrate"``.
- ``"molecule_formal_charges"``: Allowed molecular charges; typically set to ``[0]`` for neutral structures.
- ``"structures_to_exclude"``: List of structures that cause kernel errors and are thus excluded.
- ``"center_molecule"``: Whether to center the molecule in the unit cell (recommended).
- ``"add_full_component"``: Analyzes complete components in the unit cell along with fragments.
- ``"proposed_vectors_n_max"``: Maximum value for each component of a crystallographic vector, suggested value is ``5``.

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
- ``"smarts"``: SMARTS notation representing the chemical structure of the fragment.
- ``"species"``: List of atomic species corresponding to the atoms in the fragment.
- ``"coordinates"``: Positions of the atoms in the fragment in any coordinate system. These will be automatically converted to space-fixed reference coordinates by the ``create_reference_fragments.py`` script.
- ``"mass"``: List of atomic masses for each atom in the fragment.
- ``"atoms_to_align"``: Specifies which atoms in the fragment to use for alignment. It designates specific atoms within the fragment for orientation synchronization with a corresponding fragment identified in a crystal structure. This approach is particularly useful for fragments that exhibit indistinguishable, mirror-image formations, such as oxygens in a structure like [#6]S(=O)(=O)[NH2], where traditional SMARTS representation may fall short. Accepts:
  
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

  - Detailed data for each contact includes the type (vdW or H-bond), length, line of sight verification, and vectors related to central and contact fragments in both Cartesian and spherical coordinates. 
    Each contact can appear in the data file up to 8 times, corresponding to the 8 possible combinations generated by the `central atom` (2 options), the `central fragment` (2 options), and the `contact fragment` (2 options).
    For example, in the ``ACSALA24`` structure from the CSD database, a close contact forms between atoms :math:`\ce{C1}` and :math:`\ce{C2}`. 
    Atom :math:`\ce{C1}` is common to both the benzene and carboxylic acid fragments, while atom :math:`\ce{C2}` is common to the benzene ring and the ester fragment.
    As a result, the contact between these two atoms appears 8 times in the contact data file as follows: ::

        str_id   label1 label2  spec1  spec2  hbond               central_fragment               contact_fragment   ...
        ...         ...    ...    ...    ...    ...                            ...                            ...
        ACSALA24     C1     C2      C      C  False                        benzene                        benzene   ...
        ACSALA24     C1     C2      C      C  False                        benzene       ester_aromatic-aliphatic   ...
        ACSALA24     C1     C2      C      C  False                carboxylic_acid                        benzene   ...
        ACSALA24     C1     C2      C      C  False                carboxylic_acid       ester_aromatic-aliphatic   ...
        ACSALA24     C2     C1      C      C  False                        benzene                        benzene   ...
        ACSALA24     C2     C1      C      C  False       ester_aromatic-aliphatic                        benzene   ...
        ACSALA24     C2     C1      C      C  False                        benzene                carboxylic_acid   ...
        ACSALA24     C2     C1      C      C  False       ester_aromatic-aliphatic                carboxylic_acid   ...

    In the default post-extraction data analysis tool, special filters are applied to avoid using duplicate records in terms of the central and contact fragments. The contact :math:`\ce{C1}-\ce{C2}` however, is considered different compared to :math:`\ce{C2}-\ce{C1}` since relative position of the contact atom to the central fragment in the inertia frame is unique for each central-contact fragment pair.


- **Hydrogen Bond Data**:

  - For each H-bond, the algorithm determines the donor and acceptor atoms, bond length, donor-acceptor distance, bond angle, and line of sight status.

Finally, all data gathered is written to output files, completing the data extraction process.

The Data Extraction Output Files
--------------------------------
The data extraction process generates four different types of data files. Each file type is prefixed with the ``data_prefix`` defined in the input file, and their contents are as follows:

Contact Data Files
^^^^^^^^^^^^^^^^^^
File name: ``*_contact_data.txt``

This file contains all the information regarding close contacts within the structures:

- **Structure ID** (``str_id``).
- **Atom labels and species** (``label1``, ``label2``, ``spec1``, ``spec2``) for the atoms forming the contact.
- **Hydrogen bond participation** (``hbond``): Indicates if the contact is part of a hydrogen bond.
- **Fragments involved**: The fragment for the central atom and the contact atom (``central_fragment``, ``contact_fragment``).
- **Contact length** (``length``) and **verification of line of sight status** (``in_los``).
- **Atom coordinates**: 

  - (``x1``, ``y1``, ``z1``) for the central atom and.
  - (``x2``, ``y2``, ``z2``) for the contact atom.
- **Cartesian bond vectors** to the center of mass of the central group:

  - (``bvx1``, ``bvy1``, ``bvz1``) for the central atom and.
  - (``bvx2``, ``bvy2``, ``bvz2``) for the contact atom.
- **Reference system Cartesian bond vectors**:

  - (``bvx1_ref``, ``bvy1_ref``, ``bvz1_ref``) for the central atom and.
  - (``bvx2_ref``, ``bvy2_ref``, ``bvz2_ref``) for the contact atom.
- **Spherical coordinates bond vectors** for the contact atom (``r2``, ``theta2``, ``phi2``).

Fragment Data Files
^^^^^^^^^^^^^^^^^^^
File name: ``*_fragment_data.txt``

This file gathers details about the fragments in the structure:

- **Structure ID** (``str_id``).
- **Fragment name** (``fragment``) and **fragment coordinates** (``x``, ``y``, ``z``; ``u``, ``v``, ``w``).
- **Principal axes of inertia components**:

  - (``e1_x``, ``e1_y``, ``e1_z``).
  - (``e2_x``, ``e2_y``, ``e2_z``).
  - (``e3_x``, ``e3_y``, ``e3_z``).
- **Minimum distances of principal planes of inertia** to reference cell points (``d1``, ``d2``, ``d3``)
- **Normal vectors to principal axes** in crystallographic coordinates:

  - (``e1_u``, ``e1_v``, ``e1_w``).
  - (``e2_u``, ``e2_v``, ``e2_w``).
  - (``e3_u``, ``e3_v``, ``e3_w``).
- **Vectors closest to being perpendicular** to each principal axis and respective angles:

  - (``W11_u``, ``W11_v``, ``W11_w``, ``ang_11``) and (``W12_u``, ``W12_v``, ``W12_w``, ``ang_12``) for the first axis.
  - (``W21_u``, ``W21_v``, ``W21_w``, ``ang_21``) and (``W22_u``, ``W22_v``, ``W22_w``, ``ang_22``) for the second axis.
  - (``W31_u``, ``W31_v``, ``W31_w``, ``ang_31``) and (``W32_u``, ``W32_v``, ``W32_w``, ``ang_32``) for the third axis.
- **Number of atoms** in the fragment (``n_at``) and detailed atomic data:

  - **Cartesian coordinates, fractional coordinates, and minimum distances to the ZZP plane family for each atom** (``at_x``, ``at_y``, ``at_z``; ``at_u``, ``at_v``, ``at_w``; ``dzzp_min``)

Hydrogen Bond Data Files
^^^^^^^^^^^^^^^^^^^^^^^^
File name: ``*_hbond_data.txt``

This file includes comprehensive information about hydrogen bonds:

- **Structure ID** (``str_id``).
- **Labels and species** for the donor, hydrogen, and acceptor atoms (``labelD``, ``labelH``, ``labelA``; ``specD``, ``specH``, ``specA``).
- **Hydrogen bond metrics**:

  - Length (``length``).
  - Donor-acceptor distance (``DA_dis``).
  - Angle (``angle``)
  
- **Line of sight status** (``in_los``).
- **Atom coordinates**:

  - Donor (``xD``, ``yD``, ``zD``).
  - Hydrogen (``xH``, ``yH``, ``zH``).
  - Acceptor (``xA``, ``yA``, ``zA``).

Structure Data Files
^^^^^^^^^^^^^^^^^^^^
File name: ``*_structure_data.txt``

This file records comprehensive metrics and properties of the crystal structure:

- **Structure ID** (``str_id``), **space group** (``sg``), and **Z** and **Z'** values (``Z``, ``Z_pr``).
- **Chemical formula** (``formula``) and **species composition** (``species``).
- **Cell dimensions**:

  - Scaled (``a_sc``, ``b_sc``, ``c_sc``).
  - Actual (``a``, ``b``, ``c``).
  - Angles (``alpha``, ``beta``, ``gamma``).
- **Cell volume** (``volume``) and **density** (``density``).
- **Van der Waals free volume** (``vdWFV``) and **solvent accessible surface** (``SAS``).
- **Energy components**:

  - **Total lattice energy** (``E_tot``).
  - **Electrostatic energy** (``E_el``).
  - **Van der Waals energy** (``E_vdW``), including **attractive** (``E_vdW_at``) and **repulsive** (``E_vdW_rep``) components.
  - **Hydrogen bond energy** (``E_hb``), including **attractive** (``E_hb_at``) and **repulsive** (``E_hb_rep``) components.