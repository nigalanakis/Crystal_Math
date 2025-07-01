crystal_analyzer module
=======================

.. automodule:: crystal_analyzer
   :members:
   :undoc-members:
   :show-inheritance:

CrystalAnalyzer Class
---------------------

.. autoclass:: crystal_analyzer.CrystalAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

   The main orchestration class for the CSA pipeline. This class coordinates all five stages
   of the analysis workflow and manages the flow of data between components.

   **Key Responsibilities:**

   * Pipeline orchestration and stage management
   * Configuration validation and setup
   * Resource management and cleanup
   * Error handling and recovery
   * Progress monitoring and logging

   **Usage Example:**

   .. code-block:: python

      from crystal_analyzer import CrystalAnalyzer
      from csa_config import load_config

      # Load configuration
      config = load_config('analysis_config.json')
      
      # Initialize analyzer
      analyzer = CrystalAnalyzer(extraction_config=config)
      
      # Run complete pipeline
      analyzer.extract_data()

Methods
-------

extract_data()
~~~~~~~~~~~~~~

.. automethod:: crystal_analyzer.CrystalAnalyzer.extract_data

   Executes the complete five-stage CSA pipeline:

   1. **Family Extraction** - Query CSD for structure families
   2. **Similarity Clustering** - Group similar crystal packings  
   3. **Representative Selection** - Choose optimal structures
   4. **Data Extraction** - Extract detailed structural data
   5. **Feature Engineering** - Compute advanced descriptors

   Each stage can be enabled/disabled via the configuration file.

Private Methods
~~~~~~~~~~~~~~~

_extract_refcode_families()
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: crystal_analyzer.CrystalAnalyzer._extract_refcode_families

   Queries the CSD and groups structures into families based on refcode prefixes.
   
   **Returns:**
   
   * CSV file with family mappings
   * Statistics on family sizes and composition

_cluster_refcode_families()
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: crystal_analyzer.CrystalAnalyzer._cluster_refcode_families

   Performs packing similarity clustering within each family using CCDC algorithms.
   
   **Process:**
   
   1. Validates structures against filter criteria
   2. Computes 3D packing similarity for all pairs
   3. Builds similarity graphs and identifies clusters
   4. Outputs clustered family assignments

_extract_unique_structures()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: crystal_analyzer.CrystalAnalyzer._extract_unique_structures

   Selects one representative structure per cluster using the vdWFV metric.
   
   **Selection Criteria:**
   
   * Minimum van der Waals free volume (1 - packing coefficient)
   * Lexicographic tie-breaking for identical values
   * Structure quality validation

_extract_structure_data()
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: crystal_analyzer.CrystalAnalyzer._extract_structure_data

   Extracts detailed molecular and crystal data for selected representatives.
   
   **Data Extracted:**
   
   * Atomic coordinates, labels, and properties
   * Bond connectivity and rotatability
   * Intermolecular contacts and hydrogen bonds
   * Crystal parameters and symmetry operations

_post_extraction_process()
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: crystal_analyzer.CrystalAnalyzer._post_extraction_process

   Computes advanced features and descriptors using GPU acceleration.
   
   **Features Computed:**
   
   * Fragment identification and properties
   * Geometric descriptors (angles, torsions, planarity)
   * Contact mapping and interaction analysis
   * Statistical order parameters

Private Attributes
~~~~~~~~~~~~~~~~~~

extraction_config : ExtractionConfig
    Configuration object controlling pipeline behavior

csd_ops : CSDOperations
    Handler for CSD database operations

extractor : StructureDataExtractor
    Component for raw data extraction

data_dir : pathlib.Path
    Directory for intermediate and output files

Error Handling
--------------

The CrystalAnalyzer includes comprehensive error handling:

**Validation Errors**
   Configuration validation failures, missing files, invalid parameters

**Database Errors**
   CSD connectivity issues, license problems, corrupted entries

**Resource Errors**
   Insufficient memory, disk space, or GPU resources

**Processing Errors**
   Structure validation failures, computation errors, file I/O issues

All errors are logged with detailed context information to facilitate debugging.

Configuration Dependencies
---------------------------

The CrystalAnalyzer requires a properly configured ExtractionConfig object:

.. code-block:: python

   {
     "extraction": {
       "data_directory": "./output",
       "data_prefix": "analysis",
       "actions": {
         "get_refcode_families": true,
         "cluster_refcode_families": true,
         "get_unique_structures": true,
         "get_structure_data": true,
         "post_extraction_process": true
       },
       "filters": {
         "target_z_prime_values": [1],
         "crystal_type": ["homomolecular"],
         "molecule_weight_limit": 500.0,
         "target_species": ["C", "H", "N", "O"]
       },
       "extraction_batch_size": 32,
       "post_extraction_batch_size": 16
     }
   }

Performance Considerations
--------------------------

**Memory Usage**
   Peak memory usage occurs during post-extraction processing and scales with:
   
   * Batch size settings
   * Structure complexity (atoms, contacts)
   * GPU memory availability

**Processing Time**
   Pipeline duration depends on:
   
   * Dataset size (number of families/structures)
   * Similarity clustering complexity
   * Available computational resources

**Storage Requirements**
   Output file sizes scale with:
   
   * Number of selected structures
   * Average structure complexity
   * Feature completeness

**Optimization Tips**
   
   * Use GPU acceleration for stages 4-5
   * Optimize batch sizes for available memory
   * Use SSD storage for HDF5 operations
   * Monitor resource usage during processing

See Also
--------

:doc:`../core/csa_config` : Configuration management
:doc:`../extraction/csd_operations` : CSD database operations  
:doc:`../extraction/structure_data_extractor` : Raw data extraction
:doc:`../extraction/structure_post_extraction_processor` : Feature engineering