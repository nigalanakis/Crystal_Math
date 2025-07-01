csa_config module
================

.. automodule:: csa_config
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Management for CSA Pipeline
------------------------------------------

The ``csa_config`` module provides robust configuration management for the Crystal Structure Analysis pipeline through the ``ExtractionConfig`` dataclass and associated loading utilities.

ExtractionConfig Class
----------------------

.. autoclass:: csa_config.ExtractionConfig
   :members:
   :undoc-members:
   :show-inheritance:

   Configuration dataclass controlling all aspects of the CSA extraction pipeline.

   **Core Configuration Parameters:**

   * **data_directory** (:obj:`Path`) - Base directory for all extraction outputs
   * **data_prefix** (:obj:`str`) - Filename prefix for generated files  
   * **actions** (:obj:`Dict[str, bool]`) - Pipeline stage enable/disable flags
   * **filters** (:obj:`Dict[str, Any]`) - Structure filtering and validation criteria
   * **extraction_batch_size** (:obj:`int`) - Batch size for raw data extraction
   * **post_extraction_batch_size** (:obj:`int`) - Batch size for feature computation

   **Pipeline Actions Control:**

   The ``actions`` dictionary controls which pipeline stages execute:

   .. code-block:: python

      actions = {
          "get_refcode_families": True,      # Stage 1: Family extraction
          "cluster_refcode_families": True,  # Stage 2: Similarity clustering  
          "get_unique_structures": True,     # Stage 3: Representative selection
          "get_structure_data": True,        # Stage 4: Raw data extraction
          "post_extraction_process": True    # Stage 5: Feature engineering
      }

   **Filter Criteria Examples:**

   Quality filters ensure reliable structural data:

   .. code-block:: python

      filters = {
          "target_z_prime_values": [1],           # Z' constraint
          "crystal_type": ["homomolecular"],      # Single molecule type
          "molecule_weight_limit": 500.0,         # Dalton upper limit
          "target_species": ["C", "H", "N", "O"], # Allowed elements
          "min_resolution": 1.5,                  # Angstrom resolution
          "max_r_factor": 0.05,                   # R-factor quality
          "exclude_disorder": True,               # Structural quality
          "exclude_polymers": True,
          "exclude_solvates": True
      }

   **Performance Tuning:**

   Batch sizes should be optimized for available hardware:

   .. code-block:: python

      # For systems with 16GB+ GPU memory
      extraction_batch_size = 64
      post_extraction_batch_size = 32
      
      # For systems with 8GB GPU memory  
      extraction_batch_size = 32
      post_extraction_batch_size = 16

   .. automethod:: from_json

      Load configuration from JSON file's "extraction" section.

      **JSON Structure Expected:**

      .. code-block:: json

         {
           "extraction": {
             "data_directory": "./analysis_output",
             "data_prefix": "my_analysis",
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

      **Validation Performed:**

      * File existence and readability
      * Valid JSON syntax
      * Presence of "extraction" section
      * Parameter type validation
      * Path conversion for data_directory

      **Returns:**
         :obj:`ExtractionConfig` instance with validated parameters

      **Raises:**
         * :obj:`FileNotFoundError` - Configuration file not found
         * :obj:`KeyError` - Missing "extraction" section  
         * :obj:`json.JSONDecodeError` - Invalid JSON syntax

Configuration Loading Functions
-------------------------------

.. autofunction:: csa_config.load_config

   Primary entry point for loading CSA configurations.

   **Usage Pattern:**

   .. code-block:: python

      from csa_config import load_config
      from crystal_analyzer import CrystalAnalyzer

      # Load configuration
      config = load_config('my_analysis.json')
      
      # Initialize analyzer with config
      analyzer = CrystalAnalyzer(extraction_config=config)
      
      # Execute pipeline
      analyzer.extract_data()

   **Configuration Templates:**

   CSA provides template configurations for common use cases:

   * ``templates/pharmaceutical.json`` - Drug crystal analysis
   * ``templates/materials.json`` - Materials science applications  
   * ``templates/organic.json`` - General organic crystal analysis
   * ``templates/high_throughput.json`` - Large-scale screening

   **Parameters:**
      * **config_path** (:obj:`Union[str, Path]`) - Path to JSON configuration file

   **Returns:**
      :obj:`ExtractionConfig` instance ready for pipeline execution

   **Raises:**
      * :obj:`FileNotFoundError` - Configuration file not found
      * :obj:`KeyError` - Missing "extraction" section
      * :obj:`json.JSONDecodeError` - Invalid JSON syntax

Configuration Validation
-------------------------

**Pre-Flight Validation**

The configuration system performs comprehensive validation at load time:

.. code-block:: python

   try:
       config = load_config('analysis.json')
       print("✓ Configuration valid")
   except FileNotFoundError:
       print("✗ Configuration file not found")
   except KeyError as e:
       print(f"✗ Missing configuration section: {e}")
   except json.JSONDecodeError as e:
       print(f"✗ Invalid JSON syntax: {e}")

**Field Validation**

Each configuration parameter is validated for:

* **Type correctness** - String, number, boolean, array types
* **Required presence** - Essential fields must be specified
* **Value ranges** - Numeric parameters within valid bounds
* **Path validity** - Directory paths must be accessible

**Common Configuration Errors**

**Missing Required Fields:**

.. code-block:: text

   KeyError: 'extraction' section missing in config.json

*Solution:* Ensure JSON contains top-level "extraction" object

**Invalid Path Specifications:**

.. code-block:: text

   FileNotFoundError: Config file not found: /invalid/path/config.json

*Solution:* Verify file paths and permissions

**Type Mismatches:**

.. code-block:: text

   TypeError: Expected int for extraction_batch_size, got str

*Solution:* Check numeric fields are not quoted in JSON

Examples
--------

**Basic Pharmaceutical Analysis:**

.. code-block:: python

   from csa_config import load_config
   from crystal_analyzer import CrystalAnalyzer

   # Load pharmaceutical-focused configuration
   config = load_config('templates/pharmaceutical.json')
   
   # Customize for specific drug class
   config.filters.update({
       "target_species": ["C", "H", "N", "O", "S", "Cl"],
       "molecule_weight_limit": 800.0,
       "target_z_prime_values": [1, 2]
   })
   
   # Run analysis
   analyzer = CrystalAnalyzer(extraction_config=config)
   analyzer.extract_data()

**High-Throughput Materials Screening:**

.. code-block:: python

   # Load high-throughput template
   config = load_config('templates/high_throughput.json')
   
   # Optimize for speed over completeness
   config.actions.update({
       "cluster_refcode_families": False,  # Skip clustering for speed
       "post_extraction_process": False    # Skip advanced features
   })
   
   # Increase batch sizes for throughput
   config.extraction_batch_size = 128
   
   # Execute streamlined pipeline
   analyzer = CrystalAnalyzer(extraction_config=config)
   analyzer.extract_data()

**Custom Local CIF Analysis:**

.. code-block:: python

   # Configure for local CIF files
   config = load_config('templates/organic.json')
   
   # Disable CSD querying stages
   config.actions.update({
       "get_refcode_families": False,
       "cluster_refcode_families": False,
       "get_unique_structures": False
   })
   
   # Point to local CIF directory
   config.filters["structure_list"] = ["cif", "/path/to/cif/files"]
   
   # Process local structures
   analyzer = CrystalAnalyzer(extraction_config=config)
   analyzer.extract_data()

See Also
--------

:doc:`../core/crystal_analyzer` : Main pipeline orchestration
:doc:`../core/csa_main` : Command-line interface
:doc:`../getting_started/configuration` : Configuration guide