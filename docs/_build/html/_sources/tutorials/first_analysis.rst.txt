Your First Analysis
==================

.. note::
   **Duration**: ~45 minutes | **Prerequisites**: CSA installation complete
   
   **Downloads**: :download:`benzene_tutorial.json </_downloads/benzene_tutorial.json>` | :download:`benzene_tutorial_families.csv </_downloads/benzene_tutorial_families.csv>` | :download:`tutorial_validation.py </_downloads/tutorial_validation.py>`

Welcome to your first hands-on experience with Crystal Structure Analysis (CSA)! This tutorial will guide you through the complete five-stage CSA pipeline using a focused dataset of benzene structures from the CSD.

Learning Objectives
-------------------

By the end of this tutorial, you will:

* Successfully execute all five stages of the CSA pipeline
* Understand the data flow between pipeline stages
* Navigate and interpret CSA output files
* Validate results and troubleshoot common issues
* Customize basic configuration parameters for your research

Prerequisites
-------------

Before starting, ensure you have:

* ✅ CSA installed and tested (:doc:`../getting_started/installation`)
* ✅ Valid CCDC license and CSD database access
* ✅ At least 8GB available disk space
* ✅ Basic familiarity with command line and Python

.. tip::
   
   If you encounter issues, check the :ref:`troubleshooting section <first-analysis-troubleshooting>` at the bottom of this tutorial.

Tutorial Overview
-----------------

We'll analyze benzene structures from the CSD to demonstrate CSA's complete workflow. This tutorial uses a pre-defined family list to focus on a manageable dataset perfect for learning:

* **Focused dataset**: Benzene family (BENZEN) structures from CSD
* **Clear relationships**: All structures share the benzene core
* **Manageable size**: ~25-30 structures for reasonable processing time
* **Well-characterized**: Extensively studied polymorphic system

The five pipeline stages will transform our initial family list into rich, analysis-ready data:

1. **Family Extraction** - Load pre-defined benzene family refcodes (skipped for focused analysis)
2. **Similarity Clustering** - Group benzene structures by crystal packing similarity  
3. **Representative Selection** - Choose optimal structures from each cluster
4. **Data Extraction** - Extract atomic coordinates, bonds, and properties
5. **Feature Engineering** - Compute geometric descriptors and contact maps

Understanding the CSA Pipeline
------------------------------

Before diving into the tutorial, it's important to understand how CSA actually works:

**Stage 1: Family Extraction (`get_refcode_families`)**
   - Queries the entire CSD database to find all refcode families
   - **No filters applied** - this stage simply catalogs what's available
   - Output: Complete list of family_id and refcode pairs
   - For focused studies, users can provide a pre-made CSV instead

**Stage 2: Similarity Clustering (`cluster_refcode_families`)**
   - Groups structures within each family by 3D packing similarity
   - **Applies most filters** (except `structure_list`) to validate structures
   - Uses CCDC packing similarity algorithms
   - Output: Cluster assignments for each valid structure

**Stage 3: Representative Selection (`get_unique_structures`)**
   - Selects one representative per cluster using vdWFV (van der Waals Fit Volume)
   - Chooses structure closest to cluster median packing density
   - Output: List of unique representative structures

**Stage 4: Data Extraction (`get_structure_data`)**
   - Extracts detailed structural data from CSD or local CIF files
   - **`structure_list` filter determines source**: "csd-unique" (default) or "cif"
   - Processes representatives into HDF5 format
   - Output: Raw structural data with coordinates, bonds, contacts

**Stage 5: Feature Engineering (`post_extraction_process`)**
   - Computes advanced descriptors and geometric features
   - GPU-accelerated tensor operations
   - Output: Analysis-ready feature datasets

Step 1: Setup and Configuration
-------------------------------

Create Tutorial Directory
~~~~~~~~~~~~~~~~~~~~~~~~~

First, let's set up a dedicated workspace for this tutorial:

.. code-block:: bash

   # Create tutorial directory
   mkdir csa_first_analysis
   cd csa_first_analysis
   
   # Create subdirectories for organization
   mkdir configs
   mkdir scripts
   mkdir results

Download Tutorial Files
~~~~~~~~~~~~~~~~~~~~~~

For this focused tutorial, we'll use a pre-defined family list for the benzene family. Create a file named ``results/benzene_tutorial_families.csv`` with the following content:

.. code-block:: text

   family_id,refcode
   BENZEN,BENZEN
   BENZEN,BENZEN01
   BENZEN,BENZEN02
   BENZEN,BENZEN03
   BENZEN,BENZEN04
   BENZEN,BENZEN05
   BENZEN,BENZEN06
   BENZEN,BENZEN07
   BENZEN,BENZEN08
   BENZEN,BENZEN09
   BENZEN,BENZEN10
   BENZEN,BENZEN11
   BENZEN,BENZEN12
   BENZEN,BENZEN13
   BENZEN,BENZEN14
   BENZEN,BENZEN15
   BENZEN,BENZEN16
   BENZEN,BENZEN17
   BENZEN,BENZEN18
   BENZEN,BENZEN19
   BENZEN,BENZEN20
   BENZEN,BENZEN21
   BENZEN,BENZEN22
   BENZEN,BENZEN23
   BENZEN,BENZEN24
   BENZEN,BENZEN25
   BENZEN,BENZEN26
   BENZEN,BENZEN27
   BENZEN,BENZEN28

.. note::
   
   This CSV has the exact format that CSA expects: `family_id,refcode` with the benzene family containing all available BENZEN refcodes from the CSD.

Create Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~

Create a file named ``configs/benzene_tutorial.json`` with the following configuration:

.. code-block:: json

   {
     "extraction": {
       "data_directory": "../benzene_tutorial/",
       "data_prefix": "benzene_tutorial",
       "actions": {
         "get_refcode_families": false,
         "cluster_refcode_families": true,
         "get_unique_structures": true,
         "get_structure_data": true,
         "post_extraction_process": true
       },
       "filters": {
         "structure_list": ["csd-unique"],
         "crystal_type": ["homomolecular"],
         "target_species": ["C", "H"],
         "target_space_groups": ["P21/c","Pbca"],
         "target_z_prime_values": [0.5],
         "molecule_weight_limit": 100.0,
         "molecule_formal_charges": [0],
         "unique_structures_clustering_method": "vdWFV",
       },
       "extraction_batch_size": 32,
       "post_extraction_batch_size": 32
     }
   }

Configuration Explanation
~~~~~~~~~~~~~~~~~~~~~~~~~

Let's understand the key parameters in our configuration:

.. list-table:: Key Configuration Parameters
   :header-rows: 1
   :widths: 25 35 40

   * - Parameter
     - Value
     - Purpose
   * - ``get_refcode_families``
     - ``false``
     - Skip CSD-wide family extraction (using pre-made list)
   * - ``structure_list``
     - ``["csd-unique"]``
     - Use CSD database (not local CIF files)
   * - ``crystal_type``
     - ``["homomolecular"]``
     - Single molecular species crystals
   * - ``target_species``
     - ``["C", "H"]``
     - Simple hydrocarbons only
   * - ``target_space_groups``
     - ``["P21/c", "Pbca"]``
     - Use only the two availabe space groups for the known benzene structures
   * - ``target_z_prime_values``
     - ``[0.5]``
     - The availabe ``Z'`` value for the known benzene structures: 0.5 molecules per asymmetric unit
   * - ``molecule_weight_limit``
     - ``200.0``
     - Focus on benzene (78 Da) and simple derivatives
   * - ``molecule_formal_charges``
     - ``[0]``
     - Neutral molecules
   * - ``unique_structures_clustering_method``
     - ``vdWFV``
     - Metric to select unique structure from a cluster

.. note::
   
   These parameters create a focused, high-quality dataset perfect for learning CSA fundamentals. The filters are applied during clustering, not during family extraction.

Step 2: Pipeline Execution
--------------------------

Running the Complete Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now let's execute the CSA pipeline with our configuration:

.. code-block:: bash

   # Navigate to CSA installation directory
   cd /path/to/crystal-structure-analysis
   
   # Run the pipeline (adjust path to your tutorial directory)
   python src/csa_main.py --config /path/to/csa_first_analysis/configs/benzene_tutorial.json

Expected Progress Output
~~~~~~~~~~~~~~~~~~~~~~~

You should see output similar to this:

.. code-block:: text

   2025-05-04 17:21:30,846 - root - INFO - Loading configuration from csa_config.json
   2025-05-04 17:21:30,846 - root - INFO - Starting extraction step...
   2025-05-04 17:21:30,846 - crystal_analyzer - INFO - Starting data extraction pipeline...
   2025-05-04 17:21:30,846 - crystal_analyzer - INFO - Clustering refcode families...
   2025-05-04 17:21:56,171 - csd_operations - INFO - Saved clustered families to ..\benzene_tutorial\benzene_tutorial_refcode_families_clustered.csv
   2025-05-04 17:21:56,171 - crystal_analyzer - INFO - Refcode families clustered into 23 groups.
   2025-05-04 17:21:56,171 - crystal_analyzer - INFO - Selecting unique structures …
   2025-05-04 17:21:58,029 - csd_operations - INFO - Saved unique structures to ..\benzene_tutorial\benzene_tutorial_refcode_families_unique.csv
   2025-05-04 17:21:58,029 - crystal_analyzer - INFO - Unique structures selected: 2 structures across 1 families
   2025-05-04 17:21:58,029 - crystal_analyzer - INFO - Extracting detailed structure data into ..\benzene_tutorial\benzene_tutorial.h5 …
   2025-05-04 17:21:58,029 - structure_data_extractor - INFO - Overwriting existing HDF5 file: ..\benzene_tutorial\benzene_tutorial.h5
   2025-05-04 17:21:58,037 - structure_data_extractor - INFO - 2 structures to extract (batch size 1000)
   2025-05-04 17:21:58,037 - structure_data_extractor - INFO - Extracting batch 1 (size 2)
   2025-05-04 17:21:59,893 - structure_data_extractor - INFO - Raw data extraction complete; HDF5 file closed.
   2025-05-04 17:21:59,893 - crystal_analyzer - INFO - Detailed structure data extracted and saved to ..\benzene_tutorial\benzene_tutorial.h5
   2025-05-04 17:21:59,893 - structure_post_extraction_processor - INFO - Removing existing processed file: ..\benzene_tutorial\benzene_tutorial_processed.h5
   2025-05-04 17:21:59,906 - structure_post_extraction_processor - INFO - Found 2 structures to process.
   2025-05-04 17:21:59,906 - structure_post_extraction_processor - INFO - Processing structures 1 to 2
   2025-05-04 17:22:00,292 - structure_post_extraction_processor - INFO - Post-extraction fast processing complete.
   2025-05-04 17:22:00,292 - crystal_analyzer - INFO - Data extraction completed in 0:00:29.445523
   2025-05-04 17:22:00,292 - root - INFO - Data extraction completed successfully.

Performance Expectations
~~~~~~~~~~~~~~~~~~~~~~~~

Expected performance for this tutorial:

**Stage 1 (Not performed)**

**Stage 2 (<2 minutes)**
    Groups structures with similar crystal packing

**Stage 3 (<1 minute)**
    Picks the best representative from each cluster

**Stage 4 (<1 minute)**
    Extracts atomic coordinates and basic properties

**Stage 5 (<1 minute)**
    Computes advanced molecular descriptors

Step 3: Exploring the Results
-----------------------------

Output File Structure
~~~~~~~~~~~~~~~~~~~~

After successful completion, your results directory should contain:

.. code-block:: text

   results/
   ├── benzene_tutorial_families.csv              # Pre-made family list (input)
   ├── benzene_tutorial_clustered_families.csv    # Stage 2 output
   ├── benzene_tutorial_unique_structures.csv     # Stage 3 output
   ├── benzene_tutorial_structures.h5             # Stage 4 output
   └── benzene_tutorial_structures_processed.h5   # Stage 5 output

Understanding CSV Outputs
~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Input Family List**

.. code-block:: python

   import pandas as pd
   
   # Load and examine the input family list
   families_df = pd.read_csv('../benzene_tutorial/benzene_tutorial_refcode_families.csv')
   print(f"Input structures: {len(families_df)}")
   print(f"Families: {families_df['family_id'].nunique()}")
   
   # Show the family structure
   print(families_df.head(5))

Expected output:

.. code-block:: text

   Input structures: 29
   Families: 1
   
     family_id   refcode
   0    BENZEN    BENZEN
   1    BENZEN  BENZEN01
   2    BENZEN  BENZEN02
   3    BENZEN  BENZEN03
   4    BENZEN  BENZEN04
   5    BENZEN  BENZEN05
   6    BENZEN  BENZEN06
   7    BENZEN  BENZEN07
   8    BENZEN  BENZEN08
   9    BENZEN  BENZEN09

**2. Clustered Families (Stage 2)**

.. code-block:: python

   # Load clustering results
   clustered_df = pd.read_csv('../benzene_tutorial/benzene_tutorial_refcode_families_clustered..csv')
   print(f"Structures after filtering: {len(clustered_df)}")
   print(f"Total clusters formed: {clustered_df['cluster_id'].nunique()}")
   
   # Analyze cluster sizes
   cluster_sizes = clustered_df.groupby('cluster_id').size()
   print(f"Average cluster size: {cluster_sizes.mean():.2f}")
   print(f"Largest cluster: {cluster_sizes.max()} structures")
   print(f"Cluster size distribution:")
   print(cluster_sizes.value_counts().sort_index())
   
Expected output:

.. code-block:: text

   Structures after filtering: 23
   Total clusters formed: 2
   Average cluster size: 11.50
   Largest cluster: 16 structures
   Cluster size distribution:
   7     1
   16    1

**3. Representative Structures (Stage 3)**

.. code-block:: python

   # Load final structure selection
   unique_df = pd.read_csv('../benzene_tutorial/benzene_tutorial_refcode_families_unique.csv')
   print(f"Representative structures selected: {len(unique_df)}")
   
   # Show selected representatives
   print("Selected representative structures:")
   print(unique_df[['family_id', 'refcode']].to_string(index=False))
   
Expected output:

.. code-block:: text

   Representative structures selected: 2
   Selected representative structures:
   family_id  refcode
      BENZEN BENZEN22
      BENZEN BENZEN24
	  
Congratulations!
================

🎉 **Your first CSA data extraction is complete!** 

You have successfully:

✅ **Executed the complete CSA pipeline** from clustering to feature engineering
✅ **Generated analysis-ready datasets** with 2 representative benzene structures  
✅ **Created HDF5 files** containing atomic coordinates, molecular descriptors, and contact maps
✅ **Understood the data flow** between all five pipeline stages
✅ **Learned to interpret** CSV outputs and validate results

What You've Accomplished
------------------------

Your tutorial has produced:

* **2 representative benzene structures** selected from 23 valid CSD entries
* **Complete structural data** including atomic coordinates and bond connectivity  
* **Advanced molecular descriptors** like fragment properties and shape parameters
* **Intermolecular contact maps** identifying hydrogen bonds and close contacts
* **Analysis-ready HDF5 datasets** optimized for computational analysis

Next Steps: Analyzing Your Data
-------------------------------

Now that you have working CSA datasets, it's time to explore and analyze your results:

**Start with Data Access**
   📖 :doc:`../user_guide/basic_analysis` → **"Accessing Your Data"** section
   
   Learn how to load and navigate your HDF5 files, extract crystal properties, and understand the data structure CSA has created.

**Explore Analysis Workflows**  
   📊 :doc:`../user_guide/basic_analysis` → **"Essential Analysis Workflows"** section
   
   Discover practical analysis patterns including property distributions, fragment analysis, and contact network exploration.

**Recommended Learning Path**

1. **Immediate next step**: :ref:`Accessing Your Data <accessing-your-data>` to load and inspect your benzene dataset
2. **Then explore**: :ref:`Crystal Property Analysis <crystal-property-analysis>` to visualize your results  
3. **Advanced analysis**: :ref:`Fragment Analysis <fragment-analysis>` to study benzene molecular shapes
4. **Finally try**: :ref:`Contact Analysis <contact-analysis>` to map intermolecular interactions

**Ready for More?**

* **Try different chemical systems** → Modify your configuration to study other molecular families
* **Scale up your analysis** → Remove size restrictions and analyze larger datasets
* **Explore domain-specific tutorials** → :doc:`../tutorials/organic_chemistry` for hydrocarbon-specific workflows
* **Learn advanced configuration** → :doc:`../user_guide/configuration` for research-optimized setups

Welcome to the CSA community! 🚀 You're now ready to tackle real crystallographic research questions with confidence.

