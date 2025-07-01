Quickstart Guide
================

This guide will get you running your first CSA analysis in under 15 minutes.

Overview
--------

The Crystal Structure Analysis (CSA) pipeline consists of five main stages:

1. **Family Extraction** - Query CSD for structure families
2. **Similarity Clustering** - Group similar crystal packings
3. **Representative Selection** - Choose optimal structures
4. **Data Extraction** - Extract detailed structural data
5. **Feature Engineering** - Compute advanced descriptors

Prerequisites
-------------

Before starting, ensure you have:

- ✅ CSA installed (see :doc:`installation`)
- ✅ CCDC license and CSD database access
- ✅ GPU access (recommended but optional)

Step 1: Configuration Setup
---------------------------

Create Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a file named ``my_first_analysis.json``:

.. code-block:: json

    {
      "extraction": {
        "data_directory": "./csa_output",
        "data_prefix": "organic_crystals",
        "actions": {
          "get_refcode_families": true,
          "cluster_refcode_families": true,
          "get_unique_structures": true,
          "get_structure_data": true,
          "post_extraction_process": true
        },
        "filters": {
          "target_z_prime_values": [1],
          "target_space_groups": [],
          "crystal_type": ["homomolecular"],
          "molecule_formal_charges": [0],
          "molecule_weight_limit": 500.0,
          "target_species": ["C", "H", "N", "O"],
          "structure_list": ["csd-unique"]
        },
        "extraction_batch_size": 32,
        "post_extraction_batch_size": 16
      }
    }

Configuration Explained
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Parameter
     - Description
     - Example Value
   * - ``data_directory``
     - Output directory for all files
     - ``"./csa_output"``
   * - ``data_prefix``
     - Prefix for generated files
     - ``"organic_crystals"``
   * - ``target_z_prime_values``
     - Z' values to include
     - ``[1]`` (single molecule/asymmetric unit)
   * - ``crystal_type``
     - Crystal types to analyze
     - ``["homomolecular"]``
   * - ``molecule_weight_limit``
     - Max molecular weight (Da)
     - ``500.0``
   * - ``target_species``
     - Allowed chemical elements
     - ``["C", "H", "N", "O"]``
   * - ``extraction_batch_size``
     - Structures per GPU batch
     - ``32``

Step 2: Run Your First Analysis
-------------------------------

Basic Execution
~~~~~~~~~~~~~~~

.. code-block:: bash

    cd /path/to/crystal-structure-analysis
    python src/csa_main.py --config my_first_analysis.json

With Custom Logging
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python src/csa_main.py --config my_first_analysis.json 2>&1 | tee analysis.log

Expected Output
~~~~~~~~~~~~~~~

You should see output similar to::

    2024-01-15 10:30:15 - crystal_analyzer - INFO - Starting data extraction pipeline...
    2024-01-15 10:30:16 - csd_operations - INFO - Extracting refcode families into DataFrame...
    2024-01-15 10:30:45 - csd_operations - INFO - Extracted 45,823 structures across 12,456 families
    2024-01-15 10:31:15 - csd_operations - INFO - Clustering refcode families...
    2024-01-15 10:35:30 - csd_operations - INFO - Refcode families clustered into 8,234 groups
    2024-01-15 10:35:45 - csd_operations - INFO - Selecting unique structures...
    2024-01-15 10:36:12 - csd_operations - INFO - Unique structures selected: 8,234 structures across 8,234 families
    2024-01-15 10:36:15 - structure_data_extractor - INFO - 8234 structures to extract (batch size 32)
    2024-01-15 10:36:20 - structure_data_extractor - INFO - Extracting batch 1 (size 32)
    ...
    2024-01-15 11:45:30 - structure_data_extractor - INFO - Raw data extraction complete; HDF5 file closed
    2024-01-15 11:45:35 - structure_post_extraction_processor - INFO - Found 8234 structures to process
    ...
    2024-01-15 12:30:45 - structure_post_extraction_processor - INFO - Post-extraction fast processing complete
    2024-01-15 12:30:46 - crystal_analyzer - INFO - Data extraction completed in 2:00:31

Step 3: Explore Your Results
----------------------------

Generated Files
~~~~~~~~~~~~~~~

After completion, check your output directory::

    ls -la csa_output/

You should see::

    organic_crystals_refcode_families.csv          # Initial CSD query results
    organic_crystals_refcode_families_clustered.csv # Similarity clusters
    organic_crystals_refcode_families_unique.csv    # Selected representatives
    organic_crystals.h5                            # Raw structural data
    organic_crystals_processed.h5                  # Processed features

Quick Data Inspection
~~~~~~~~~~~~~~~~~~~~~

Use this Python script to inspect your results:

.. code-block:: python

    import h5py
    import pandas as pd
    import numpy as np

    # Load the processed data
    with h5py.File('csa_output/organic_crystals_processed.h5', 'r') as f:
        # Get basic statistics
        refcodes = f['refcode_list'][...].astype(str)
        n_structures = len(refcodes)
        
        print(f"Successfully processed {n_structures:,} crystal structures")
        print(f"First 5 refcodes: {refcodes[:5].tolist()}")
        
        # Check crystal properties
        z_prime = f['z_prime'][...]
        cell_volumes = f['cell_volume'][...]
        
        print(f"\nCrystal Properties:")
        print(f"  Z' values: {np.unique(z_prime)}")
        print(f"  Cell volume range: {cell_volumes.min():.1f} - {cell_volumes.max():.1f} Ų")
        print(f"  Mean cell volume: {cell_volumes.mean():.1f} Ų")
        
        # Check molecular composition
        n_atoms = f['n_atoms'][...]
        print(f"\nMolecular Composition:")
        print(f"  Atoms per molecule: {n_atoms.min()} - {n_atoms.max()}")
        print(f"  Mean atoms per molecule: {n_atoms.mean():.1f}")
        
        # Check fragment analysis
        n_fragments = f['n_fragments'][...]
        print(f"\nFragment Analysis:")
        print(f"  Fragments per structure: {n_fragments.min()} - {n_fragments.max()}")
        print(f"  Mean fragments per structure: {n_fragments.mean():.1f}")
        
        # Check intermolecular contacts
        n_contacts = f['inter_cc_n_contacts'][...]
        contacts_nonzero = n_contacts[n_contacts > 0]
        print(f"\nIntermolecular Contacts:")
        print(f"  Structures with contacts: {len(contacts_nonzero):,} ({len(contacts_nonzero)/len(n_contacts)*100:.1f}%)")
        if len(contacts_nonzero) > 0:
            print(f"  Contacts per structure: {contacts_nonzero.min()} - {contacts_nonzero.max()}")
            print(f"  Mean contacts: {contacts_nonzero.mean():.1f}")

Sample Analysis
~~~~~~~~~~~~~~~

Try this simple analysis to explore crystal packing efficiency:

.. code-block:: python

    import h5py
    import matplotlib.pyplot as plt
    import numpy as np

    # Load processed data
    with h5py.File('csa_output/organic_crystals_processed.h5', 'r') as f:
        packing_coefficients = f['packing_coefficient'][...]
        cell_volumes = f['cell_volume'][...]
        n_atoms = f['n_atoms'][...]

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Packing efficiency distribution
    ax1.hist(packing_coefficients, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Packing Coefficient')
    ax1.set_ylabel('Number of Structures')
    ax1.set_title('Crystal Packing Efficiency Distribution')
    ax1.axvline(packing_coefficients.mean(), color='red', linestyle='--', 
               label=f'Mean: {packing_coefficients.mean():.3f}')
    ax1.legend()

    # Volume vs. molecular size
    ax2.scatter(n_atoms, cell_volumes, alpha=0.5)
    ax2.set_xlabel('Number of Atoms per Molecule')
    ax2.set_ylabel('Unit Cell Volume (Ų)')
    ax2.set_title('Cell Volume vs. Molecular Size')

    plt.tight_layout()
    plt.savefig('csa_output/quick_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Analysis complete! Plot saved to: csa_output/quick_analysis.png")

Step 4: Access Advanced Features
--------------------------------

Explore Fragment Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import h5py
    import numpy as np

    with h5py.File('csa_output/organic_crystals_processed.h5', 'r') as f:
        # Fragment center-of-mass coordinates
        structure_idx = 0  # First structure
        n_frags = f['n_fragments'][structure_idx]
        
        if n_frags > 0:
            # Get fragment formulas
            formulas = f['fragment_formula'][structure_idx]
            print(f"Structure has {n_frags} fragments:")
            for i, formula in enumerate(formulas):
                print(f"  Fragment {i+1}: {formula}")
            
            # Get fragment planarity scores
            planarity = f['fragment_planarity_score'][structure_idx]
            planarity_vals = planarity[:n_frags]  # First n_frags values
            print(f"\nPlanarity scores: {planarity_vals}")

Analyze Intermolecular Contacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import h5py

    with h5py.File('csa_output/organic_crystals_processed.h5', 'r') as f:
        # Find structures with hydrogen bonds
        structures_with_hbonds = []
        
        for i in range(min(100, len(f['refcode_list']))):  # Check first 100
            n_hbonds = f['inter_hb_n_hbonds'][i]
            if n_hbonds > 0:
                refcode = f['refcode_list'][i].decode()
                structures_with_hbonds.append((refcode, n_hbonds))
        
        print(f"Found {len(structures_with_hbonds)} structures with hydrogen bonds:")
        for refcode, n_hbonds in structures_with_hbonds[:10]:  # Show first 10
            print(f"  {refcode}: {n_hbonds} H-bonds")

Common Issues and Solutions
---------------------------

Issue: "No structures found"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Cause**: Filters too restrictive

**Solution**: Relax filters (increase weight limit, add more elements, etc.)

Issue: GPU out of memory
~~~~~~~~~~~~~~~~~~~~~~~~
**Cause**: Batch size too large

**Solution**: Reduce ``extraction_batch_size`` and ``post_extraction_batch_size``

Issue: Very slow processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Cause**: CPU-only processing or large dataset

**Solutions**: 
- Enable GPU acceleration
- Reduce dataset size for testing
- Use smaller batch sizes

Issue: HDF5 file corruption
~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Cause**: Interrupted processing or disk space issues

**Solution**: Delete partial files and restart with more disk space

Next Steps
----------

Now that you've completed your first CSA analysis:

1. **Explore the Data**: Use the analysis scripts above to understand your results
2. **Try Different Filters**: Experiment with different chemical systems
3. **Learn Advanced Features**: Read the :doc:`../user_guide/index` for detailed explanations
4. **Customize Analysis**: Check out :doc:`../examples/index` for specific use cases
5. **Optimize Performance**: Review :doc:`../technical_details/performance` 

Getting Help
------------

If you encounter issues:

- **Check logs**: Look for error messages in the console output
- **Verify configuration**: Ensure JSON syntax is correct
- **Test with small datasets**: Use restrictive filters to process fewer structures
- **Consult documentation**: Read the relevant sections for your use case
- **Report bugs**: Submit issues with full error traces and configuration files

Continue to the :doc:`../user_guide/index` to learn more about CSA's capabilities and advanced usage patterns.