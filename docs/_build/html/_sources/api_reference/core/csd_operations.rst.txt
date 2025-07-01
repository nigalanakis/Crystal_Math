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
      FAM001       ABINIK
      FAM001       ABINIK01  
      FAM001       ABINIK02
      FAM002       ACETAC
      FAM002       ACETAC01
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
      FAM001,ABINIK
      FAM001,ABINIK01
      FAM002,ACETAC
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
      FAM001       ABINIK      1
      FAM001       ABINIK01    1  
      FAM001       ABINIK02    2
      FAM002       ACETAC      1
      FAM002       ACETAC01    1
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
   * **Resolution limits** - X-ray diffraction quality
   * **R-factor thresholds** - Refinement quality indicators
   * **Completeness requirements** - Data collection completeness
   * **Temperature ranges** - Experimental condition constraints

   **Chemical Filters:**
   * **Element restrictions** - Allowed atomic species
   * **Molecular weight limits** - Size constraints
   * **Z' value constraints** - Asymmetric unit requirements
   * **Crystal type requirements** - Homomolecular vs. solvated

   **Structural Filters:**
   * **Disorder exclusion** - Remove disordered structures
   * **Polymer exclusion** - Exclude polymeric materials
   * **Solvate handling** - Include/exclude solvated structures

   **Example Filter Configuration:**

   .. code-block:: python

      filters = {
          "min_resolution": 1.5,           # Å
          "max_r_factor": 0.05,            # 5% max R-factor
          "target_species": ["C","H","N","O"], # Organic only
          "molecule_weight_limit": 500.0,   # 500 Da limit
          "target_z_prime_values": [1],     # Z' = 1 only
          "exclude_disorder": True,         # No disorder
          "exclude_polymers": True,         # No polymers
          "exclude_solvates": True          # No solvates
      }

   **Returns:**
      :obj:`bool` indicating if structure passes all validation criteria

Representative Selection Methods
--------------------------------

.. automethod:: csd_operations.CSDOperations.get_unique_structures

   **Select Representative Structures from Clusters**

   Chooses one optimal representative structure from each cluster using the vdWFV (van der Waals Fit Volume) metric.

   **Selection Algorithm:**

   The vdWFV method selects the structure with the most typical packing density within each cluster:

   .. code-block:: python

      # For each cluster, compute vdWFV for all members
      vdwfv_values = {}
      for refcode in cluster:
          entry = reader.entry(refcode)
          vdwfv_values[refcode] = entry.crystal.vdw_fit_volume
      
      # Select structure closest to cluster median
      median_vdwfv = np.median(list(vdwfv_values.values()))
      representative = min(vdwfv_values.items(), 
                          key=lambda x: abs(x[1] - median_vdwfv))[0]

   **Parameters:**
      * **method** (:obj:`str`) - Selection method ("vdWFV" only supported)

   **Selection Criteria:**
      * **Typicality** - Representative of cluster packing behavior
      * **Quality** - Prefers high-quality experimental data
      * **Completeness** - Avoids structures with missing data

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
      FAM001,ABINIK01
      FAM002,ACETAC
      FAM003,BENZAC02
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

Usage Examples
--------------

**Basic CSD Operations Workflow**

.. code-block:: python

   from csd_operations import CSDOperations
   from pathlib import Path

   # Initialize CSD operations
   csd_ops = CSDOperations(
       data_directory=Path("./analysis_output"),
       data_prefix="my_analysis"
   )

   # Step 1: Extract refcode families
   families_df = csd_ops.get_refcode_families_df()
   csd_ops.save_refcode_families_csv(families_df)

   # Step 2: Filter by family size  
   filtered_df = csd_ops.filter_families_by_size(families_df, min_size=3)

   # Step 3: Cluster families by similarity
   filters = {
       "target_z_prime_values": [1],
       "molecule_weight_limit": 500.0,
       "target_species": ["C", "H", "N", "O"]
   }
   clustered_df = csd_ops.cluster_families(filters)

   # Step 4: Select unique representatives
   unique_df = csd_ops.get_unique_structures(method="vdWFV")

**Custom Similarity Settings**

.. code-block:: python

   from csd_operations import SimilaritySettings, CSDOperations

   # Configure custom similarity parameters
   strict_settings = SimilaritySettings(
       distance_tolerance=0.1,     # Stricter distance matching
       angle_tolerance=10.0,       # Stricter angle matching
       packing_shell_size=20,      # Larger comparison shell
       ignore_hydrogen_positions=False  # Include H positions
   )

   # Initialize with custom settings
   csd_ops = CSDOperations(
       data_directory="./output",
       data_prefix="strict_analysis"
   )
   
   # Apply custom settings to similarity engine
   csd_ops.similarity_engine.settings = strict_settings

**High-Throughput Processing**

.. code-block:: python

   import multiprocessing as mp
   from csd_operations import CSDOperations

   # Configure for maximum parallelization
   n_cores = mp.cpu_count() - 2  # Leave 2 cores free
   
   # Process large datasets efficiently
   csd_ops = CSDOperations("./large_analysis", "high_throughput")
   
   # Use relaxed filters for speed
   fast_filters = {
       "target_z_prime_values": [1, 2],
       "molecule_weight_limit": 1000.0,
       "min_resolution": 2.0,  # Relaxed resolution
       "max_r_factor": 0.10    # Relaxed R-factor
   }
   
   # Execute clustering with optimized settings
   clustered = csd_ops.cluster_families(fast_filters)

**Quality Control and Validation**

.. code-block:: python

   # Implement comprehensive quality control
   quality_filters = {
       "min_resolution": 1.0,           # High resolution only
       "max_r_factor": 0.03,            # Strict R-factor
       "min_completeness": 0.95,        # 95% data completeness
       "exclude_disorder": True,         # No disorder
       "exclude_polymers": True,         # No polymers
       "exclude_solvates": True,         # No solvates
       "temperature_range": [90, 120],   # Low-temperature data
       "target_species": ["C","H","N","O","S","Cl","F"]  # Common elements
   }

   # Validate structures before clustering
   csd_ops = CSDOperations("./quality_analysis", "high_quality")
   
   for family_id, refcodes in families.items():
       valid_structures = []
       for refcode in refcodes:
           if csd_ops._check_structure(refcode, quality_filters):
               valid_structures.append(refcode)
       
       if len(valid_structures) >= 2:
           print(f"Family {family_id}: {len(valid_structures)} valid structures")

Performance Optimization
------------------------

**Memory Management**

.. code-block:: python

   # Configure for large datasets
   import gc
   
   def memory_efficient_clustering(csd_ops, filters, batch_size=100):
       """Process families in memory-efficient batches."""
       
       families_df = csd_ops.get_refcode_families_df()
       family_groups = families_df.groupby('family_id')
       
       all_results = []
       for i, (family_id, group) in enumerate(family_groups):
           if i % batch_size == 0:
               gc.collect()  # Periodic garbage collection
           
           # Process single family
           result = csd_ops._process_single_family(
               (family_id, group['refcode'].tolist(), filters)
           )
           all_results.append(result)
       
       return all_results

**Parallel Processing Tuning**

.. code-block:: python

   import os
   from concurrent.futures import ProcessPoolExecutor

   # Optimize worker count based on system resources
   optimal_workers = min(
       os.cpu_count() - 2,        # Leave CPU headroom
       32,                        # Reasonable upper limit
       len(family_groups) // 4    # At least 4 families per worker
   )

   print(f"Using {optimal_workers} parallel workers")

**Disk I/O Optimization**

.. code-block:: python

   # Use SSD storage for temporary files
   fast_storage = Path("/tmp/csd_analysis")  # RAM disk or SSD
   fast_storage.mkdir(exist_ok=True)
   
   csd_ops = CSDOperations(
       data_directory=fast_storage,
       data_prefix="temp_analysis"
   )
   
   # Move final results to permanent storage
   import shutil
   final_location = Path("./permanent_storage")
   shutil.move(fast_storage, final_location)

See Also
--------

:doc:`../core/crystal_analyzer` : Pipeline orchestration
:doc:`../extraction/structure_data_extractor` : Raw data extraction  
:doc:`../validation/csd_structure_validator` : Structure validation
:doc:`../processing/geometry_utils` : Geometric calculations