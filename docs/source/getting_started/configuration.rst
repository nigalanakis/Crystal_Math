Configuration
=============

CSA uses JSON configuration files to control all aspects of the analysis pipeline. This guide covers the complete configuration system, from basic setups to advanced customizations.

.. note::
   
   Configuration files are validated at startup. Invalid configurations will cause CSA to exit with detailed error messages.

Configuration Structure
-----------------------

All CSA configurations follow this top-level structure:

.. code-block:: json

    {
      "extraction": {
        "data_directory": "./output",
        "data_prefix": "analysis_name",
        "actions": { ... },
        "filters": { ... },
        "extraction_batch_size": 32,
        "post_extraction_batch_size": 16
      }
    }

The ``extraction`` section contains all pipeline settings. Other top-level sections may be added in future versions.

Required Parameters
-------------------

data_directory
~~~~~~~~~~~~~~

**Type**: ``string``  
**Description**: Output directory for all generated files  
**Example**: ``"./csa_output"``  

This directory will be created if it doesn't exist. All intermediate and final output files will be stored here:

- ``csv/`` - Family extractions and clustering results
- ``structures/`` - HDF5 datasets with raw and processed data
- ``logs/`` - Detailed execution logs

data_prefix  
~~~~~~~~~~~

**Type**: ``string``  
**Description**: Prefix for all output filenames  
**Example**: ``"polymer_analysis"``

Results in files like:
- ``polymer_analysis_refcode_families.csv``
- ``polymer_analysis_structures.h5``
- ``polymer_analysis_structures_processed.h5``

Pipeline Control
----------------

actions
~~~~~~~

Controls which stages of the five-stage pipeline to execute:

.. code-block:: json

    "actions": {
      "get_refcode_families": true,
      "cluster_refcode_families": true, 
      "get_unique_structures": true,
      "get_structure_data": true,
      "post_extraction_process": true
    }

.. list-table:: Pipeline Actions
   :widths: 30 50 20
   :header-rows: 1

   * - Action
     - Description
     - Default
   * - ``get_refcode_families``
     - Query CSD and extract refcode families
     - ``true``
   * - ``cluster_refcode_families``
     - Group similar crystal packings
     - ``true``
   * - ``get_unique_structures``
     - Select representative structures
     - ``true``  
   * - ``get_structure_data``
     - Extract atomic coordinates and bonds
     - ``true``
   * - ``post_extraction_process``
     - Compute advanced features and descriptors
     - ``true``

CSD Filtering
-------------

filters
~~~~~~~

The ``filters`` object controls which structures are included from the CSD:

.. code-block:: json

    "filters": {
      "target_z_prime_values": [1],
      "target_space_groups": [],
      "crystal_type": ["homomolecular"],
      "molecule_formal_charges": [0],
      "molecule_weight_limit": 1000.0,
      "target_species": ["C", "H", "N", "O"],
      "structure_list": ["csd-unique"]
    }

Structure Selection
~~~~~~~~~~~~~~~~~~~

.. list-table:: Structure Filters
   :widths: 25 35 20 20
   :header-rows: 1

   * - Filter
     - Description
     - Type
     - Default
   * - ``target_z_prime_values``
     - Number of molecules per asymmetric unit
     - ``array[int]``
     - ``[1]``
   * - ``target_space_groups``
     - Allowed crystallographic space groups
     - ``array[string]``
     - ``[]`` (all)
   * - ``crystal_type``
     - Crystal composition type
     - ``array[string]``
     - ``["homomolecular"]``
   * - ``structure_list``
     - Source database/collection
     - ``array[string]``
     - ``["csd-unique"]``

Chemical Filters
~~~~~~~~~~~~~~~~

.. list-table:: Chemical Filters  
   :widths: 25 35 20 20
   :header-rows: 1

   * - Filter
     - Description
     - Type
     - Default
   * - ``molecule_formal_charges``
     - Allowed molecular charges
     - ``array[int]``
     - ``[0]``
   * - ``molecule_weight_limit``
     - Maximum molecular weight (Da)
     - ``float``
     - ``1000.0``
   * - ``target_species``
     - Required chemical elements
     - ``array[string]``
     - ``[]`` (all)

Performance Settings
--------------------

Batch Sizes
~~~~~~~~~~~

Control memory usage and processing speed:

.. code-block:: json

    "extraction_batch_size": 32,
    "post_extraction_batch_size": 16

.. list-table:: Batch Size Guidelines
   :widths: 30 25 25 20
   :header-rows: 1

   * - Hardware
     - Extraction Batch
     - Post-Processing Batch
     - Notes
   * - CPU only
     - 8-16
     - 4-8
     - Conservative settings
   * - GPU (8GB)
     - 32-64
     - 16-32
     - Standard configuration
   * - GPU (16GB+)
     - 64-128
     - 32-64
     - High-performance setup
   * - GPU (24GB+)
     - 128-256
     - 64-128
     - Maximum throughput

.. warning::
   Batch sizes that are too large will cause out-of-memory errors. Start with smaller values and increase gradually.

Example Configurations
----------------------

Basic Organic Analysis
~~~~~~~~~~~~~~~~~~~~~~

Small organic molecules with standard elements:

.. code-block:: json

    {
      "extraction": {
        "data_directory": "./organic_analysis",
        "data_prefix": "organics",
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
          "molecule_formal_charges": [0],
          "molecule_weight_limit": 500.0,
          "target_species": ["C", "H", "N", "O"]
        },
        "extraction_batch_size": 32,
        "post_extraction_batch_size": 16
      }
    }

Pharmaceutical Analysis
~~~~~~~~~~~~~~~~~~~~~~~

Drug-like molecules including halogens and sulfur:

.. code-block:: json

    {
      "extraction": {
        "data_directory": "./pharma_analysis", 
        "data_prefix": "pharmaceuticals",
        "filters": {
          "target_z_prime_values": [1, 2],
          "crystal_type": ["homomolecular"],
          "molecule_formal_charges": [0, 1, -1],
          "molecule_weight_limit": 800.0,
          "target_species": ["C", "H", "N", "O", "S", "F", "Cl", "Br"]
        },
        "extraction_batch_size": 64,
        "post_extraction_batch_size": 32
      }
    }

High-Throughput Screening
~~~~~~~~~~~~~~~~~~~~~~~~~

Maximum performance for large datasets:

.. code-block:: json

    {
      "extraction": {
        "data_directory": "./hts_analysis",
        "data_prefix": "high_throughput", 
        "filters": {
          "target_z_prime_values": [1],
          "crystal_type": ["homomolecular"],
          "molecule_weight_limit": 600.0
        },
        "extraction_batch_size": 128,
        "post_extraction_batch_size": 64
      }
    }

Local CIF File Analysis
~~~~~~~~~~~~~~~~~~~~~~~

Analyze your own CIF files instead of querying the CSD:

.. code-block:: json

    {
      "extraction": {
        "data_directory": "./local_cif_analysis",
        "data_prefix": "my_structures",
        "actions": {
          "get_refcode_families": false,
          "cluster_refcode_families": false,
          "get_unique_structures": false,
          "get_structure_data": true,
          "post_extraction_process": true
        },
        "filters": {
          "structure_list": ["cif", "/path/to/your/cif/files"]
        },
        "extraction_batch_size": 32,
        "post_extraction_batch_size": 16
      }
    }

Configuration Validation
-------------------------

CSA performs comprehensive validation of configuration files at startup:

Schema Validation
~~~~~~~~~~~~~~~~~

All parameters are checked for:
- **Type correctness** (string, number, array, boolean)
- **Required fields** presence  
- **Value ranges** for numeric parameters
- **Valid options** for enumerated fields

Common Validation Errors
~~~~~~~~~~~~~~~~~~~~~~~~~

**Invalid JSON Syntax**

.. code-block:: text

    ERROR: Invalid JSON in config file: Expecting ',' delimiter: line 15 column 5

*Solution*: Check for missing commas, quotes, or brackets

**Missing Required Fields**

.. code-block:: text

    ERROR: Missing required field 'data_directory' in extraction config

*Solution*: Add all required parameters to your configuration

**Invalid Parameter Values**

.. code-block:: text

    ERROR: extraction_batch_size must be between 1 and 256, got 512

*Solution*: Use values within the specified ranges

**Invalid File Paths**

.. code-block:: text

    ERROR: CIF directory '/nonexistent/path' not found

*Solution*: Ensure all file paths exist and are accessible

Configuration Inheritance
--------------------------

Default Configuration
~~~~~~~~~~~~~~~~~~~~~

CSA includes built-in defaults for all optional parameters:

.. code-block:: python

    DEFAULT_CONFIG = {
        "extraction": {
            "actions": {
                "get_refcode_families": True,
                "cluster_refcode_families": True,
                "get_unique_structures": True,
                "get_structure_data": True,
                "post_extraction_process": True
            },
            "filters": {
                "target_z_prime_values": [1],
                "target_space_groups": [],
                "crystal_type": ["homomolecular"],
                "molecule_formal_charges": [0],
                "molecule_weight_limit": 1000.0,
                "target_species": [],
                "structure_list": ["csd-unique"]
            },
            "extraction_batch_size": 32,
            "post_extraction_batch_size": 16
        }
    }

Minimal Configuration
~~~~~~~~~~~~~~~~~~~~~

The smallest valid configuration requires only the essential parameters:

.. code-block:: json

    {
      "extraction": {
        "data_directory": "./output",
        "data_prefix": "analysis"
      }
    }

All other parameters will use their default values.

Environment Variables
---------------------

Some configuration values can be overridden using environment variables:

**CSD Database Path**

.. code-block:: bash

    export CCDC_CSD_DIRECTORY="/path/to/csd/database"

**GPU Device Selection**

.. code-block:: bash

    export CUDA_VISIBLE_DEVICES="0,1"  # Use GPUs 0 and 1

**Memory Optimization**

.. code-block:: bash

    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

Best Practices
--------------

Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use descriptive prefixes**: Include date, target, or purpose in data_prefix
2. **Version your configs**: Keep configurations with your analysis results  
3. **Document modifications**: Use comments to explain non-standard settings
4. **Test with small datasets**: Validate configurations with restricted filters first

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Start conservative**: Begin with small batch sizes and increase gradually
2. **Monitor resources**: Watch GPU memory and system RAM usage during runs
3. **Profile your system**: Different hardware configurations require different settings
4. **Use appropriate filters**: More restrictive filters reduce processing time

Reproducibility
~~~~~~~~~~~~~~~

1. **Save complete configs**: Store the exact configuration used for each analysis
2. **Record software versions**: Document CSA, PyTorch, and CCDC versions
3. **Timestamp analyses**: Include analysis dates in output directories
4. **Validate inputs**: Ensure CSD versions and filter criteria are documented

Troubleshooting
---------------

Configuration Issues
~~~~~~~~~~~~~~~~~~~~

**Problem**: ``FileNotFoundError: Config file not found``

*Solution*: Check file path and ensure the configuration file exists

**Problem**: ``JSON decode error``

*Solution*: Validate JSON syntax using an online JSON validator

**Problem**: ``Invalid extraction configuration``

*Solution*: Compare against the examples in this guide and check for typos

Performance Issues
~~~~~~~~~~~~~~~~~~

**Problem**: Out of GPU memory during processing

*Solution*: Reduce batch sizes in the configuration

**Problem**: Very slow processing on CPU

*Solution*: Install GPU-enabled PyTorch and reduce batch sizes

**Problem**: Disk space errors

*Solution*: Ensure sufficient space in the data directory

Next Steps
----------

With your configuration ready:

1. **Validate syntax**: Test your JSON using online validators
2. **Start small**: Run with restrictive filters first  
3. **Monitor performance**: Watch resource usage during initial runs
4. **Scale up gradually**: Increase dataset size and batch sizes as appropriate
5. **Document settings**: Keep records of successful configurations

Continue to :doc:`quickstart` to run your first analysis, or explore :doc:`../user_guide/basic_analysis` for detailed workflow examples.