Configuration
=============

CSA uses JSON configuration files to control all aspects of the analysis pipeline. This guide covers the essential configuration concepts you need to get started quickly.

.. note::
   
   This guide covers basic configuration for getting started. For advanced research-driven configurations, see :doc:`../user_guide/configuration`.

Quick Start Configuration
-------------------------

Minimal Configuration
~~~~~~~~~~~~~~~~~~~~~

The simplest CSA configuration requires only two parameters:

.. code-block:: json

    {
      "extraction": {
        "data_directory": "./my_analysis",
        "data_prefix": "my_first_run"
      }
    }

This will use default settings for all other parameters.

Complete Basic Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For your first analysis, use this template:

.. code-block:: json

    {
      "extraction": {
        "data_directory": "./csa_output",
        "data_prefix": "organic_analysis",
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

Essential Parameters
-------------------

Required Settings
~~~~~~~~~~~~~~~~

**data_directory**
  Where CSA will save all output files
  
  Example: ``"./my_analysis_output"``

**data_prefix**
  Prefix for all generated filenames
  
  Example: ``"pharma_study"`` → ``pharma_study_structures.h5``

Pipeline Control
~~~~~~~~~~~~~~~

**actions** - Controls which pipeline stages to run:

.. code-block:: json

    "actions": {
      "get_refcode_families": true,      // Query CSD for structures
      "cluster_refcode_families": true,  // Group similar packings
      "get_unique_structures": true,     // Select representatives
      "get_structure_data": true,        // Extract coordinates
      "post_extraction_process": true    // Compute features
    }

Set any action to ``false`` to skip that stage.

Basic Filters
~~~~~~~~~~~~~

**target_z_prime_values**
  Number of molecules per asymmetric unit
  
  Common values: ``[1]`` (most structures), ``[1, 2]`` (include Z'=2)

**crystal_type**
  Type of crystal structures to include
  
  Options: ``["homomolecular"]``, ``["co-crystal"]``, ``["organometallic"]``

**molecule_formal_charges**
  Allowed molecular charges
  
  Typical: ``[0]`` (neutral), ``[0, 1, -1]`` (include ions)

**molecule_weight_limit**
  Maximum molecular weight in Daltons
  
  Examples: ``300.0`` (small molecules), ``800.0`` (larger molecules)

**target_species**
  Required chemical elements
  
  Examples:
  - ``["C", "H", "N", "O"]`` - Basic organics
  - ``["C", "H", "N", "O", "S", "F", "Cl"]`` - Pharmaceuticals
  - ``[]`` - All elements (empty array)

Performance Settings
-------------------

Batch Sizes
~~~~~~~~~~~

Control memory usage and speed:

**extraction_batch_size**
  Structures processed together during data extraction
  
  - Start with: ``32``
  - If you have lots of RAM/GPU memory: ``64`` or ``128``
  - If you get memory errors: ``16`` or ``8``

**post_extraction_batch_size**
  Structures processed together during feature computation
  
  - Start with: ``16``
  - If you have lots of RAM/GPU memory: ``32`` or ``64``
  - If you get memory errors: ``8`` or ``4``

Common Configurations
--------------------

Small Organic Molecules
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

    {
      "extraction": {
        "data_directory": "./small_organics",
        "data_prefix": "small_molecules",
        "filters": {
          "target_z_prime_values": [1],
          "crystal_type": ["homomolecular"],
          "molecule_formal_charges": [0],
          "molecule_weight_limit": 300.0,
          "target_species": ["C", "H", "N", "O"]
        },
        "extraction_batch_size": 32,
        "post_extraction_batch_size": 16
      }
    }

Drug-Like Molecules
~~~~~~~~~~~~~~~~~~

.. code-block:: json

    {
      "extraction": {
        "data_directory": "./pharmaceuticals",
        "data_prefix": "drug_molecules",
        "filters": {
          "target_z_prime_values": [1, 2],
          "crystal_type": ["homomolecular"],
          "molecule_formal_charges": [0, 1, -1],
          "molecule_weight_limit": 600.0,
          "target_species": ["C", "H", "N", "O", "S", "F", "Cl", "Br"]
        },
        "extraction_batch_size": 32,
        "post_extraction_batch_size": 16
      }
    }

Quick Test Run
~~~~~~~~~~~~~

For testing CSA quickly with a small dataset:

.. code-block:: json

    {
      "extraction": {
        "data_directory": "./test_run",
        "data_prefix": "quick_test",
        "filters": {
          "target_z_prime_values": [1],
          "crystal_type": ["homomolecular"],
          "molecule_formal_charges": [0],
          "molecule_weight_limit": 200.0,
          "target_species": ["C", "H", "N", "O"],
          "max_structures": 100
        },
        "extraction_batch_size": 16,
        "post_extraction_batch_size": 8
      }
    }

Note: ``max_structures`` limits the total number of structures processed.

Creating Your Configuration
---------------------------

Step-by-Step Process
~~~~~~~~~~~~~~~~~~~

1. **Copy a template** from the examples above
2. **Modify the basics**:
   - Change ``data_directory`` to your desired output location
   - Set ``data_prefix`` to describe your analysis
3. **Adjust filters** for your research:
   - Set appropriate molecular weight limit
   - Choose relevant chemical elements
   - Decide on charge states and Z' values
4. **Set batch sizes** based on your hardware:
   - Start with the defaults (32, 16)
   - Reduce if you get memory errors
   - Increase if you have powerful hardware
5. **Save as a .json file**

Validation and Testing
---------------------

Check Your JSON
~~~~~~~~~~~~~~~

Before running CSA, validate your JSON syntax:

1. **Use an online JSON validator** (search "JSON validator")
2. **Check for common errors**:
   - Missing commas between items
   - Missing quotes around strings
   - Mismatched brackets or braces

Test with Small Dataset
~~~~~~~~~~~~~~~~~~~~~~

Always test new configurations with a small dataset first:

.. code-block:: json

    "filters": {
      "max_structures": 50,
      // ... your other filters
    }

This limits processing to 50 structures, making testing fast.

Common Beginner Mistakes
------------------------

JSON Syntax Errors
~~~~~~~~~~~~~~~~~~

**Missing Commas**

❌ Wrong:
.. code-block:: json

    {
      "data_directory": "./output"
      "data_prefix": "analysis"
    }

✅ Correct:
.. code-block:: json

    {
      "data_directory": "./output",
      "data_prefix": "analysis"
    }

**Quotes Around Strings**

❌ Wrong:
.. code-block:: json

    {
      "target_species": [C, H, N, O]
    }

✅ Correct:
.. code-block:: json

    {
      "target_species": ["C", "H", "N", "O"]
    }

Configuration Issues
~~~~~~~~~~~~~~~~~~~

**Too Restrictive Filters**

If CSA finds no structures, your filters might be too strict:
- Increase ``molecule_weight_limit``
- Add more elements to ``target_species``
- Include more charge states or Z' values

**Memory Problems**

If you get "out of memory" errors:
- Reduce ``extraction_batch_size`` to 16 or 8
- Reduce ``post_extraction_batch_size`` to 8 or 4
- Use fewer structures for testing

**Very Slow Processing**

If CSA runs very slowly:
- Check that you have GPU acceleration enabled
- Reduce the dataset size for initial testing
- Consider using a more powerful computer

Getting Help
-----------

When Things Go Wrong
~~~~~~~~~~~~~~~~~~~

1. **Check the error message** - CSA provides detailed error information
2. **Validate your JSON** - Use an online JSON validator
3. **Try a simpler configuration** - Start with the minimal example and add complexity
4. **Test with fewer structures** - Add ``"max_structures": 50`` to your filters
5. **Check the examples** - Compare your configuration to the working examples above

Next Steps
----------

Once you have a working configuration:

1. **Run your first analysis** - Follow the :doc:`quickstart` guide
2. **Explore the results** - Learn what CSA produces
3. **Try different filters** - Experiment with different molecular systems
4. **Learn advanced configuration** - Check :doc:`../user_guide/configuration` for research-specific setups

See Also
--------

:doc:`quickstart` : Run your first analysis with your configuration
:doc:`../user_guide/configuration` : Advanced configuration strategies
:doc:`../user_guide/basic_analysis` : Understanding and analyzing your results