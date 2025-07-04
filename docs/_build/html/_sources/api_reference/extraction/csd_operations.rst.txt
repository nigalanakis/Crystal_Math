csd_operations module
====================

.. automodule:: csd_operations
   :members:
   :undoc-members:
   :show-inheritance:

Cambridge Structural Database Operations
-----------------------------------------

The ``csd_operations`` module provides high-level interfaces for interacting with the Cambridge Structural Database (CSD), including family extraction, similarity clustering, and representative structure selection.

SimilaritySettings Class
------------------------

.. autoclass:: csd_operations.SimilaritySettings
   :members:
   :undoc-members:
   :show-inheritance:

   **Configuration for Packing Similarity Comparisons**

   Dataclass controlling parameters for 3D crystal packing similarity calculations using CCDC algorithms.

   **Key Parameters:**

   * **distance_tolerance** (:obj:`float`) - Maximum deviation in atomic distances (Å)
   * **angle_tolerance** (:obj:`float`) - Maximum angular deviation (degrees)  
   * **packing_shell_size** (:obj:`int`) - Number of molecules in comparison shell
   * **ignore_hydrogen_positions** (:obj:`bool`) - Whether to ignore H-atom coordinates
   * **normalise_unit_cell** (:obj:`bool`) - Whether to normalize unit cell parameters

   **Default Configuration:**

   .. code-block:: python

      settings = SimilaritySettings(
          distance_tolerance=0.2,        # 0.2 Å distance tolerance
          angle_tolerance=20.0,          # 20° angular tolerance  
          ignore_bond_types=True,        # Ignore bond order differences
          ignore_hydrogen_counts=True,   # Ignore H-count differences
          ignore_hydrogen_positions=True,# Ignore H-position differences
          packing_shell_size=15,         # 15-molecule comparison shell
          ignore_spacegroup=True,        # Ignore space group differences
          normalise_unit_cell=True       # Normalize unit cell parameters
      )

   **Tuning Guidelines:**

   * **Strict Similarity** - Reduce distance/angle tolerances
   * **Loose Similarity** - Increase tolerances for broader clustering
   * **Performance** - Reduce packing_shell_size for faster comparisons
   * **Accuracy** - Increase packing_shell_size for more reliable comparisons

CSDOperations Class
-------------------

.. autoclass:: csd_operations.CSDOperations
   :members:
   :undoc-members:
   :show-inheritance:

   **High-Level CSD Interface for Structure Operations**

   Primary interface for querying, validating, clustering, and selecting crystal structures from the Cambridge Structural Database.

   **Core Responsibilities:**

   * **Family Extraction** - Query and organize structures into chemical families
   * **Quality Validation** - Filter structures based on experimental criteria
   * **Similarity Clustering** - Group structures by 3D packing similarity
   * **Representative Selection** - Choose optimal structures using statistical metrics
   * **Data Management** - Save intermediate results and manage file I/O

   **Attributes:**
      * **data_directory** (:obj:`Path`) - Base directory for file operations
      * **data_prefix** (:obj:`str`) - Filename prefix for all generated files
      * **reader** (:obj:`io.EntryReader`) - CCDC database connection
      * **similarity_engine** (:obj:`PackingSimilarity`) - Packing comparison engine

   .. automethod:: __init__

      **Initialize CSD Operations Handler**

      Parameters:
         * **data_directory** (:obj:`Union[str, Path]`) - Base directory for file I/O
         * **data_prefix** (:obj:`str`) - Prefix for generated filenames

      **Initialization Process:**

      .. code-block:: python

         # Set up file paths and directories
         self.data_directory = Path(data_directory)
         self.data_prefix = data_prefix
         
         # Initialize CSD connection
         self.reader = io.EntryReader("CSD")
         
         # Set up packing similarity engine
         self.similarity_engine = PackingSimilarity()

      **Directory Structure Created:**

      .. code-block:: text

         data_directory/
         ├── {prefix}_refcode_families.csv
         ├── {prefix}_refcode_families_clustered.csv  
         ├── {prefix}_refcode_families_unique.csv
         └── structures/
             ├── REFCODE01.cif
             ├── REFCODE02.cif
             └── ...

Family Extraction Methods
--------------------------

.. automethod:: csd_operations.CSDOperations.get_refcode_families_df

   **Extract Structure Families from CSD**

   Queries the CSD to organize structures into families based on chemical similarity and refcode relationships.

   **Family Organization:**

   Structures are grouped by:
   * **Chemical connectivity** - Same molecular graph
   * **Refcode patterns** - Related experimental studies  
   * **Publication relationships** - Same research group/journal

   **Returns:**
      :obj:`pandas.DataFrame` with columns:
         * **family_id** - Unique identifier for each chemical family
         * **refcode** - CSD refcode for individual structures

   **Example Output:**

   .. code-block:: text

      family_id    refcode
      ACSALA       ACSALA
      ACSALA       ACSALA01  
      ACSALA       ACSALA02
      ...
      BENZEN       BENZEN
      BENZEN       BENZEN01
      BENZEN       BENZEN02
      ...

.. automethod:: csd_operations.CSDOperations.save_refcode_families_csv

   **Save Family Assignments to CSV**

   Writes refcode family assignments to disk for persistence and downstream processing.

   **Parameters:**
      * **df** (:obj:`pandas.DataFrame`, optional) - DataFrame to save; if None, generates new one
      * **filename** (:obj:`Union[str, Path]`, optional) - Output path; if None, uses default naming

   **Default File Path:**

   .. code-block:: text

      {data_directory}/{data_prefix}_refcode_families.csv

   **CSV Format:**

   .. code-block:: text

      family_id,refcode
      ACSALA,ACSALA
      ACSALA,ACSALA01
      ...
      BENZEN,BENZEN
      BENZEN,BENZEN01
      ...

.. automethod:: csd_operations.CSDOperations.filter_families_by_size

   **Filter Families by Member Count**

   Removes families with insufficient members for meaningful clustering analysis.

   **Parameters:**
      * **df** (:obj:`pandas.DataFrame`) - Family assignments DataFrame
      * **min_size** (:obj:`int`) - Minimum family size (default: 2)

   **Filtering Logic:**

   .. code-block:: python

      # Count members per family
      family_counts = df['family_id'].value_counts()
      
      # Keep only families with sufficient members
      valid_families = family_counts[family_counts >= min_size].index
      filtered_df = df[df['family_id'].isin(valid_families)]

   **Use Cases:**
      * **Statistical significance** - Ensure meaningful clustering
      * **Computational efficiency** - Focus on families with multiple structures
      * **Quality control** - Remove singleton families

   **Returns:**
      :obj:`pandas.DataFrame` with filtered family assignments

Clustering Methods
------------------

.. automethod:: csd_operations.CSDOperations.cluster_families

   **Perform Packing Similarity Clustering**

   Groups structures within each family based on 3D crystal packing similarity using CCDC algorithms.

   **Clustering Workflow:**

   1. **Load Families** - Read refcode family assignments
   2. **Parallel Processing** - Distribute families across CPU cores
   3. **Structure Validation** - Apply quality filters to each structure
   4. **Similarity Computation** - Calculate pairwise packing similarities
   5. **Graph Construction** - Build similarity graphs with threshold cutoffs
   6. **Cluster Identification** - Find connected components as clusters
   7. **Result Aggregation** - Combine results from all families

   **Parameters:**
      * **filters** (:obj:`Dict[str, Any]`) - Structure validation criteria

   **Similarity Algorithm:**

   .. code-block:: python

      # For each pair of structures in a family
      similarity = PackingSimilarity.compare(
          crystal1=entry1.crystal,
          crystal2=entry2.crystal,
          distance_tolerance=0.2,
          angle_tolerance=20.0,
          packing_shell_size=15
      )
      
      # Similarity values range from 0 (dissimilar) to 1 (identical)
      if similarity > threshold:
          graph.add_edge(refcode1, refcode2)

   **Cluster Output:**

   .. code-block:: text

      family_id    refcode     cluster_id
      ACSALA       ACSALA      1
      ACSALA       ACSALA01    1
      ACSALA       ACSALA02    1
      ...
      ACSALA       ACSALA13    2
      ACSALA       ACSALA15    2
      ACSALA       ACSALA17    2  
      ...
      ACSALA       ACSALA23    3  
      ACSALA       ACSALA24    3
      ...
      BENZEN       BENZEN      1
      BENZEN       BENZEN01    1
      BENZEN       BENZEN02    1
      ...
      BENZEN       BENZEN03    2
      BENZEN       BENZEN04    2
      BENZEN       BENZEN16    2
      ...

   **Performance Characteristics:**
      * **CPU Parallelization** - Uses multiple cores for family processing
      * **Memory Efficiency** - Processes families independently
      * **Scalability** - Linear scaling with number of families

   **Returns:**
      :obj:`pandas.DataFrame` with clustered family assignments

   **Raises:**
      * :obj:`FileNotFoundError` - If refcode families CSV is missing
      * :obj:`RuntimeError` - If clustering fails for any family

.. automethod:: csd_operations.CSDOperations._check_structure

   **Validate Structure Against Filter Criteria**

   Applies comprehensive quality filters to determine structure suitability for analysis.

   **Parameters:**
      * **identifier** (:obj:`str`) - CSD refcode to validate
      * **filters** (:obj:`Dict`) - Validation criteria dictionary
      * **entry** (:obj:`io.Entry`, optional) - Pre-loaded CSD entry

   **Validation Categories:**

   **Quality Filters:**
   * **Completeness requirements** - Data collection completeness

   **Chemical Filters:**
   * **Element restrictions** - Allowed atomic species
   * **Molecular weight limits** - Size constraints
   * **Z' value constraints** - Asymmetric unit requirements
   * **Crystal type requirements** - Homomolecular vs. solvated

   **Structural Filters:**
   * **Disorder exclusion** - Remove disordered structures
   * **Polymer exclusion** - Exclude polymeric materials

   **Example Filter Configuration:**

   .. code-block:: python

      filters = {
          "structure_list": ["csd-unique"],     # Use unique structures in CSD
          "crystal_type": ["homomolecular"],    # Homomolecular structures only
          "target_species": ["C", "H"],         # Hydrocarbons only
          "target_space_groups": ["P1", "P-1"], # Triclinic structures only
          "target_z_prime_values": [1],         # Z' = 1 only
          "molecule_weight_limit": 300.0,       # Small molecules
          "molecule_formal_charges": [0],       # Neutral molecules
          "unique_structures_clustering_method": "vdWFV", # Use van der Waals free volume to select unique structures from a cluster
      }

   **Returns:**
      :obj:`bool` indicating if structure passes all validation criteria

Representative Selection Methods
--------------------------------

.. automethod:: csd_operations.CSDOperations.get_unique_structures

   **Select Representative Structures from Clusters**

   Chooses one optimal representative structure from each cluster using the vdWFV (van der Waals Fit Volume) metric.

   **Selection Algorithm:**

   The vdWFV method selects the structure with the most typical packing density within each cluster

   **Parameters:**
      * **method** (:obj:`str`) - Selection method ("vdWFV" only supported)

   **Output Files:**
      * **CSV** - ``{prefix}_refcode_families_unique.csv``
      * **Structure Directory** - Individual CIF files for representatives

   **Returns:**
      :obj:`pandas.DataFrame` with selected representative structures

   **Raises:**
      * :obj:`FileNotFoundError` - If clustered families CSV is missing
      * :obj:`NotImplementedError` - If method other than "vdWFV" requested

.. automethod:: csd_operations.CSDOperations._save_unique_structures

   **Save Representative Structures to CSV**

   Persists the selected unique structures for downstream processing.

   **Parameters:**
      * **df** (:obj:`pandas.DataFrame`) - DataFrame with representative assignments

   **Output Format:**

   .. code-block:: text

      family_id,refcode
      ACSALA,ACSALA13
      ACSALA,ACSALA24
      ACSALA,ACSALA35
      BENZEN,BENZEN22
      BENZEN,BENZEN24
      ...

   **File Location:**

   .. code-block:: text

      {data_directory}/{data_prefix}_refcode_families_unique.csv

Utility Functions
-----------------

**_process_single_family**

.. autofunction:: csd_operations._process_single_family

   **Process Individual Family for Clustering**

   Worker function for parallel clustering of structure families.

   **Parameters:**
      * **args** (:obj:`Tuple[str, List[str], Dict]`) - (family_id, refcodes, filters)

   **Processing Steps:**
      1. **Validation** - Check each structure against filters
      2. **Similarity Matrix** - Compute all pairwise similarities
      3. **Graph Construction** - Build similarity network
      4. **Clustering** - Identify connected components
      5. **Result Packaging** - Return cluster assignments

   **Returns:**
      :obj:`Tuple[str, List[List[str]]]` - (family_id, list of clusters)

**_representative_for_cluster**

.. autofunction:: csd_operations._representative_for_cluster

   **Select Representative from Single Cluster**

   Worker function for parallel representative selection.

   **Selection Process:**
      1. **Load Structures** - Access CSD entries for cluster members
      2. **Compute Metrics** - Calculate vdWFV for each structure
      3. **Statistical Analysis** - Find cluster median vdWFV
      4. **Representative Selection** - Choose structure closest to median

   **Returns:**
      :obj:`Tuple[str, str]` - (family_id, representative_refcode)

See Also
--------

:doc:`../core/crystal_analyzer` : Pipeline orchestration
:doc:`../extraction/structure_data_extractor` : Raw data extraction  
:doc:`../validation/csd_structure_validator` : Structure validation
:doc:`../processing/geometry_utils` : Geometric calculations