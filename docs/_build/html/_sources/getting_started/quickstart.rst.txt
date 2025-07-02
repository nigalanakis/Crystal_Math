Quickstart Guide
================

This guide will get you running your first CSA analysis in under 15 minutes. You'll go from a basic configuration to analyzing crystal structure data.

Prerequisites
-------------

Before starting, ensure you have:

- ✅ CSA installed (see :doc:`installation`)
- ✅ CCDC license and CSD database access
- ✅ Virtual environment activated

.. code-block:: bash

   # Activate your CSA environment
   source csa_env/bin/activate  # macOS/Linux
   csa_env\Scripts\activate     # Windows

The Five-Stage CSA Pipeline
---------------------------

CSA transforms CSD data through five stages:

1. **Family Extraction** - Query CSD for structure families
2. **Similarity Clustering** - Group similar crystal packings  
3. **Representative Selection** - Choose optimal structures
4. **Data Extraction** - Extract detailed structural data
5. **Feature Engineering** - Compute advanced descriptors

Let's run through each stage with your first analysis.

Step 1: Create Your Configuration
---------------------------------

Create a Simple Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a file named ``my_first_analysis.json``:

.. code-block:: json

    {
      "extraction": {
        "data_directory": "./my_first_csa_run",
        "data_prefix": "organic_crystals",
        "filters": {
          "target_z_prime_values": [1],
          "crystal_type": ["homomolecular"],
          "molecule_formal_charges": [0],
          "molecule_weight_limit": 400.0,
          "target_species": ["C", "H", "N", "O"]
        },
        "extraction_batch_size": 32,
        "post_extraction_batch_size": 16
      }
    }

Configuration Explained
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Setting
     - What it does
     - Why this value?
   * - ``data_directory``
     - Where CSA saves all output files
     - Keep results organized
   * - ``data_prefix``
     - Prefix for all generated filenames
     - Identify this analysis
   * - ``target_z_prime_values: [1]``
     - Only structures with 1 molecule/asymmetric unit
     - Simplest, most common case
   * - ``crystal_type: ["homomolecular"]``
     - Only single-component crystals
     - Avoid complex multi-component systems
   * - ``molecule_weight_limit: 400.0``
     - Maximum molecular weight in Daltons
     - Focus on small-medium molecules
   * - ``target_species: ["C", "H", "N", "O"]``
     - Only basic organic elements
     - Common pharmaceutical/organic elements
   * - ``extraction_batch_size: 32``
     - Process 32 structures at once
     - Good balance of speed vs memory

Step 2: Run Your First Analysis
-------------------------------

Execute the Pipeline
~~~~~~~~~~~~~~~~~~~

Navigate to your CSA directory and run:

.. code-block:: bash

   cd /path/to/crystal-structure-analysis
   python src/csa_main.py --config my_first_analysis.json

Monitor Progress
~~~~~~~~~~~~~~~

You'll see output like this:

.. code-block:: text

   2024-07-01 10:30:15 - INFO - Starting CSA pipeline...
   2024-07-01 10:30:16 - INFO - Stage 1: Extracting refcode families from CSD...
   2024-07-01 10:30:45 - INFO - Found 15,234 structures across 8,567 families
   2024-07-01 10:31:15 - INFO - Stage 2: Clustering similar crystal packings...
   2024-07-01 10:35:30 - INFO - Created 6,123 packing clusters
   2024-07-01 10:35:45 - INFO - Stage 3: Selecting representative structures...
   2024-07-01 10:36:12 - INFO - Selected 6,123 unique structures
   2024-07-01 10:36:15 - INFO - Stage 4: Extracting structure data...
   2024-07-01 10:36:20 - INFO - Processing batch 1/192 (structures 1-32)
   ...
   2024-07-01 11:45:30 - INFO - Stage 5: Computing molecular features...
   2024-07-01 12:30:45 - INFO - Pipeline completed successfully!

**Total time**: Typically 1-2 hours for this configuration.

Understanding the Stages
~~~~~~~~~~~~~~~~~~~~~~~~

**Stage 1 (5-15 minutes)**: Queries the CSD database using your filters
**Stage 2 (10-30 minutes)**: Groups structures with similar crystal packing
**Stage 3 (1-2 minutes)**: Picks the best representative from each cluster
**Stage 4 (30-60 minutes)**: Extracts atomic coordinates and basic properties  
**Stage 5 (30-90 minutes)**: Computes advanced molecular descriptors

Step 3: Explore Your Results
----------------------------

Check Generated Files
~~~~~~~~~~~~~~~~~~~~

After completion, examine your output directory:

.. code-block:: bash

   ls -la my_first_csa_run/

You should see:

.. code-block:: text

   my_first_csa_run/
   ├── csv/
   │   ├── organic_crystals_refcode_families.csv      # Stage 1 output
   │   ├── organic_crystals_clustered_families.csv    # Stage 2 output
   │   └── organic_crystals_unique_structures.csv     # Stage 3 output
   ├── structures/
   │   ├── organic_crystals_structures.h5             # Stage 4 output
   │   └── organic_crystals_structures_processed.h5   # Stage 5 output
   └── logs/
       └── csa_extraction.log                         # Detailed logs

Quick Data Overview
~~~~~~~~~~~~~~~~~~

Use this Python script to inspect your results:

.. code-block:: python

   import h5py
   import pandas as pd
   import numpy as np

   # Basic dataset information
   with h5py.File('my_first_csa_run/structures/organic_crystals_structures_processed.h5', 'r') as f:
       refcodes = f['refcode_list'][...].astype(str)
       n_structures = len(refcodes)
       
       print(f"🎉 Successfully processed {n_structures:,} crystal structures!")
       print(f"📝 First 5 refcodes: {refcodes[:5].tolist()}")
       
       # Crystal properties overview
       space_groups = [f['space_group'][i].decode() for i in range(min(n_structures, 1000))]
       unique_sgs = set(space_groups)
       print(f"🔬 Found {len(unique_sgs)} different space groups")
       
       cell_volumes = f['cell_volume'][...]
       print(f"📏 Cell volume range: {cell_volumes.min():.1f} - {cell_volumes.max():.1f} Ų")
       
       n_atoms = f['n_atoms'][...]
       print(f"⚛️  Molecular size: {n_atoms.min()}-{n_atoms.max()} atoms (avg: {n_atoms.mean():.1f})")
       
       # Fragment analysis
       n_fragments = f['n_fragments'][...]
       print(f"🧩 Fragments per molecule: {n_fragments.min()}-{n_fragments.max()} (avg: {n_fragments.mean():.1f})")
       
       # Contact analysis
       n_contacts = f['inter_cc_n_contacts'][...]
       structures_with_contacts = np.sum(n_contacts > 0)
       print(f"🤝 {structures_with_contacts:,} structures have intermolecular contacts ({structures_with_contacts/n_structures*100:.1f}%)")

Save this as ``inspect_results.py`` and run:

.. code-block:: bash

   python inspect_results.py

Expected output:

.. code-block:: text

   🎉 Successfully processed 6,123 crystal structures!
   📝 First 5 refcodes: ['AABHTZ', 'AABHTZ01', 'AACBDE', 'AACCEF', 'AADDGH']
   🔬 Found 47 different space groups
   📏 Cell volume range: 234.5 - 2,847.3 Ų
   ⚛️ Molecular size: 8-32 atoms (avg: 18.4)
   🧩 Fragments per molecule: 1-3 (avg: 1.2)
   🤝 5,891 structures have intermolecular contacts (96.2%)

Step 4: Your First Analysis
---------------------------

Crystal Property Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

Let's analyze the crystal properties you just extracted:

.. code-block:: python

   import h5py
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd

   # Load data
   with h5py.File('my_first_csa_run/structures/organic_crystals_structures_processed.h5', 'r') as f:
       data = {
           'refcode': f['refcode_list'][...].astype(str),
           'space_group': [f['space_group'][i].decode() for i in range(len(f['refcode_list']))],
           'cell_volume': f['cell_volume'][...],
           'cell_density': f['cell_density'][...],
           'n_atoms': f['n_atoms'][...],
           'packing_coefficient': f['packing_coefficient'][...]
       }

   df = pd.DataFrame(data)

   # Create visualizations
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))

   # 1. Density distribution
   axes[0,0].hist(df['cell_density'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
   axes[0,0].set_xlabel('Crystal Density (g/cm³)')
   axes[0,0].set_ylabel('Number of Structures')
   axes[0,0].set_title('Crystal Density Distribution')
   axes[0,0].axvline(df['cell_density'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {df["cell_density"].mean():.2f}')
   axes[0,0].legend()

   # 2. Volume vs molecular size
   axes[0,1].scatter(df['n_atoms'], df['cell_volume'], alpha=0.6, color='orange')
   axes[0,1].set_xlabel('Number of Atoms')
   axes[0,1].set_ylabel('Cell Volume (Ų)')
   axes[0,1].set_title('Cell Volume vs Molecular Size')

   # 3. Top 10 space groups
   top_sgs = df['space_group'].value_counts().head(10)
   axes[1,0].barh(range(len(top_sgs)), top_sgs.values, color='lightgreen')
   axes[1,0].set_yticks(range(len(top_sgs)))
   axes[1,0].set_yticklabels(top_sgs.index)
   axes[1,0].set_xlabel('Number of Structures')
   axes[1,0].set_title('Most Common Space Groups')

   # 4. Packing efficiency
   axes[1,1].hist(df['packing_coefficient'], bins=50, alpha=0.7, color='purple', edgecolor='black')
   axes[1,1].set_xlabel('Packing Coefficient')
   axes[1,1].set_ylabel('Number of Structures')
   axes[1,1].set_title('Crystal Packing Efficiency')
   axes[1,1].axvline(df['packing_coefficient'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["packing_coefficient"].mean():.3f}')
   axes[1,1].legend()

   plt.tight_layout()
   plt.savefig('my_first_csa_run/crystal_analysis.png', dpi=300, bbox_inches='tight')
   plt.show()

   print(f"📊 Analysis complete! Plot saved to: my_first_csa_run/crystal_analysis.png")
   print(f"📈 Key findings:")
   print(f"   • Average density: {df['cell_density'].mean():.2f} g/cm³")
   print(f"   • Most common space group: {df['space_group'].mode()[0]} ({df['space_group'].value_counts().iloc[0]} structures)")
   print(f"   • Average packing efficiency: {df['packing_coefficient'].mean():.3f}")

Fragment Shape Analysis
~~~~~~~~~~~~~~~~~~~~~~

Explore molecular fragment shapes:

.. code-block:: python

   import h5py
   import matplotlib.pyplot as plt
   import numpy as np

   # Load fragment data
   fragment_data = []
   
   with h5py.File('my_first_csa_run/structures/organic_crystals_structures_processed.h5', 'r') as f:
       for i in range(min(1000, len(f['refcode_list']))):  # First 1000 structures
           refcode = f['refcode_list'][i].decode()
           n_frags = f['n_fragments'][i]
           
           if n_frags > 0:
               # Get inertia eigenvalues for shape analysis
               inertia_flat = f['fragment_inertia_eigvals'][i]
               inertia_eigvals = inertia_flat.reshape(n_frags, 3)
               
               for j in range(n_frags):
                   # Calculate shape descriptors
                   asphericity = inertia_eigvals[j, 2] - 0.5*(inertia_eigvals[j, 0] + inertia_eigvals[j, 1])
                   acylindricity = inertia_eigvals[j, 1] - inertia_eigvals[j, 0]
                   
                   fragment_data.append({
                       'refcode': refcode,
                       'asphericity': asphericity,
                       'acylindricity': acylindricity
                   })

   # Classify shapes
   shapes = []
   for frag in fragment_data:
       if frag['asphericity'] < 0.1 and frag['acylindricity'] < 0.1:
           shapes.append('spherical')
       elif frag['acylindricity'] < 0.1:
           shapes.append('oblate')
       elif frag['asphericity'] > 0.3:
           shapes.append('prolate')
       else:
           shapes.append('intermediate')

   # Plot results
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

   # Shape distribution
   shape_counts = pd.Series(shapes).value_counts()
   ax1.pie(shape_counts.values, labels=shape_counts.index, autopct='%1.1f%%')
   ax1.set_title('Molecular Fragment Shapes')

   # Shape parameter space
   asph = [frag['asphericity'] for frag in fragment_data]
   acyl = [frag['acylindricity'] for frag in fragment_data]
   
   scatter = ax2.scatter(asph, acyl, c=[{'spherical': 0, 'oblate': 1, 'prolate': 2, 'intermediate': 3}[s] for s in shapes], 
                        alpha=0.6, cmap='viridis')
   ax2.set_xlabel('Asphericity')
   ax2.set_ylabel('Acylindricity')
   ax2.set_title('Fragment Shape Parameter Space')
   
   plt.tight_layout()
   plt.savefig('my_first_csa_run/fragment_shapes.png', dpi=300, bbox_inches='tight')
   plt.show()

   print(f"🧩 Fragment shape analysis complete!")
   print(f"   • Analyzed {len(fragment_data)} molecular fragments")
   print(f"   • Shape distribution: {dict(shape_counts)}")

Step 5: Understanding Your Results
---------------------------------

What You've Created
~~~~~~~~~~~~~~~~~~

Your CSA analysis has generated:

1. **Structure Database**: 6,000+ carefully selected, non-redundant crystal structures
2. **Molecular Properties**: Comprehensive geometric and chemical descriptors
3. **Fragment Analysis**: Rigid molecular fragment identification and characterization
4. **Contact Networks**: Detailed intermolecular interaction data
5. **Crystal Properties**: Unit cell, symmetry, and packing information

Key Insights from Your Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

From this analysis, you can now investigate:

- **Packing Preferences**: Which space groups are most common for organic molecules?
- **Size-Property Relationships**: How does molecular size affect crystal density?
- **Shape Analysis**: What molecular shapes are most prevalent?
- **Packing Efficiency**: How efficiently do organic molecules pack in crystals?

Common Issues and Solutions
---------------------------

"No structures found"
~~~~~~~~~~~~~~~~~~~~

**Cause**: Filters too restrictive  
**Solution**: Increase molecular weight limit or add more chemical elements

.. code-block:: json

   "filters": {
     "molecule_weight_limit": 600.0,  // Increase from 400
     "target_species": ["C", "H", "N", "O", "S", "F"]  // Add sulfur and fluorine
   }

"Out of memory" errors
~~~~~~~~~~~~~~~~~~~~~

**Cause**: Batch sizes too large for your system  
**Solution**: Reduce batch sizes

.. code-block:: json

   "extraction_batch_size": 16,        // Reduce from 32
   "post_extraction_batch_size": 8     // Reduce from 16

"Very slow processing"
~~~~~~~~~~~~~~~~~~~~~

**Cause**: CPU-only processing  
**Solutions**:
- Enable GPU acceleration (see :doc:`installation`)
- Use smaller test dataset first
- Consider cloud computing for large analyses

Next Steps: Expanding Your Analysis
----------------------------------

Try Different Chemical Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Pharmaceutical molecules**:

.. code-block:: json

   "filters": {
     "target_species": ["C", "H", "N", "O", "S", "F", "Cl", "Br"],
     "molecule_weight_limit": 600.0
   }

**Coordination compounds**:

.. code-block:: json

   "filters": {
     "target_species": ["C", "H", "N", "O", "Fe", "Cu", "Zn"],
     "crystal_type": ["organometallic"]
   }

Explore Advanced Features
~~~~~~~~~~~~~~~~~~~~~~~~

Now that you have working CSA installation and data:

1. **Learn the full data model** - :doc:`../user_guide/data_model`
2. **Try advanced analysis workflows** - :doc:`../user_guide/basic_analysis`
3. **Explore domain-specific tutorials** - :doc:`../tutorials/index`
4. **Optimize for your research** - :doc:`../user_guide/configuration`

Scale Up Your Research
~~~~~~~~~~~~~~~~~~~~~

- **Remove size limits** for comprehensive surveys
- **Add performance optimizations** for larger datasets  
- **Integrate with your existing analysis workflows**
- **Explore machine learning applications** with your data

Congratulations!
---------------

🎉 **You've successfully completed your first CSA analysis!**

You now have:
- ✅ A working CSA installation
- ✅ Understanding of the five-stage pipeline
- ✅ Your first crystal structure dataset
- ✅ Basic analysis and visualization skills
- ✅ Knowledge to expand to your research questions

Ready for More?
--------------

Continue your CSA journey:

- **Understand your data better** → :doc:`../user_guide/basic_analysis`
- **Learn advanced configuration** → :doc:`../user_guide/configuration`  
- **Try domain-specific examples** → :doc:`../tutorials/index`
- **Explore all CSA features** → :doc:`../user_guide/index`

Welcome to the CSA community! 🚀