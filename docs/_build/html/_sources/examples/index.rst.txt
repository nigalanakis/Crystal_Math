Examples
========

Ready-to-run code examples demonstrating CSA capabilities. These practical examples show how to accomplish specific tasks and can be directly adapted for your own analyses.

.. note::
   
   All examples include complete, executable code with sample data. Copy-paste and run to see immediate results.

Quick Start Examples
-------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: ⚡ 5-Minute Quick Analysis
      :link: quick_analysis
      :link-type: doc

      Complete pipeline execution with minimal configuration.

   .. grid-item-card:: 📱 One-Liner Commands
      :link: oneliners
      :link-type: doc

      Powerful single-command workflows for common tasks.

   .. grid-item-card:: 🔧 Configuration Examples
      :link: config_examples
      :link-type: doc

      Real-world configuration files for different scenarios.

   .. grid-item-card:: 📊 Basic Visualizations
      :link: basic_plots
      :link-type: doc

      Simple plots and charts for crystal structure data.

Data Access Examples
-------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 📖 Reading HDF5 Data
      :link: reading_data
      :link-type: doc

      Efficient methods for accessing CSA datasets.

   .. grid-item-card:: 🔍 Filtering and Selection
      :link: data_filtering
      :link-type: doc

      Select subsets of structures based on criteria.

   .. grid-item-card:: 💾 Data Export
      :link: data_export
      :link-type: doc

      Export CSA data to various formats (CSV, JSON, etc.).

   .. grid-item-card:: 🔄 Data Conversion
      :link: data_conversion
      :link-type: doc

      Convert between different data representations.

Analysis Examples
----------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 📏 Geometric Analysis
      :link: geometry_examples
      :link-type: doc

      Calculate distances, angles, and geometric descriptors.

   .. grid-item-card:: 🧩 Fragment Properties
      :link: fragment_examples
      :link-type: doc

      Analyze molecular fragments and rigid bodies.

   .. grid-item-card:: 🔗 Contact Analysis
      :link: contact_examples
      :link-type: doc

      Study intermolecular interactions and packing.

   .. grid-item-card:: 📈 Statistical Analysis
      :link: statistics_examples
      :link-type: doc

      Apply statistical methods to crystal structure data.

Visualization Examples
---------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 📊 Property Distributions
      :link: distribution_plots
      :link-type: doc

      Histograms, box plots, and distribution analysis.

   .. grid-item-card:: 🗺️ Correlation Heatmaps
      :link: correlation_plots
      :link-type: doc

      Visualize relationships between crystal properties.

   .. grid-item-card:: 3️⃣ 3D Structure Plots
      :link: structure_plots
      :link-type: doc

      Interactive 3D visualization of crystal structures.

   .. grid-item-card:: 🌐 Network Diagrams
      :link: network_plots
      :link-type: doc

      Hydrogen bonding networks and contact graphs.

Integration Examples
-------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🐼 Pandas Integration
      :link: pandas_examples
      :link-type: doc

      Work with CSA data using pandas DataFrames.

   .. grid-item-card:: 🤖 Scikit-learn Integration
      :link: sklearn_examples
      :link-type: doc

      Machine learning workflows with crystal structure data.

   .. grid-item-card:: 🔢 NumPy Operations
      :link: numpy_examples
      :link-type: doc

      Efficient numerical computing with CSA datasets.

   .. grid-item-card:: 📓 Jupyter Notebooks
      :link: jupyter_examples
      :link-type: doc

      Interactive analysis workflows in notebooks.

Performance Examples
-------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: ⚡ GPU Acceleration
      :link: gpu_examples
      :link-type: doc

      Leverage GPU computing for large-scale analysis.

   .. grid-item-card:: 🔄 Batch Processing
      :link: batch_examples
      :link-type: doc

      Efficient processing of large datasets.

   .. grid-item-card:: 💾 Memory Optimization
      :link: memory_examples
      :link-type: doc

      Minimize memory usage for large analyses.

   .. grid-item-card:: ⏱️ Performance Monitoring
      :link: profiling_examples
      :link-type: doc

      Monitor and optimize analysis performance.

Example Categories
-----------------

Basic Data Operations
~~~~~~~~~~~~~~~~~~~~

**Loading and Exploring Data**

.. code-block:: python

   # Load processed HDF5 file
   import h5py
   import numpy as np
   
   with h5py.File('analysis_processed.h5', 'r') as f:
       refcodes = f['refcode_list'][...].astype(str)
       n_structures = len(refcodes)
       print(f"Dataset contains {n_structures} structures")
       
       # Quick data overview
       print(f"Available datasets: {list(f.keys())}")

**Simple Property Analysis**

.. code-block:: python

   # Analyze cell volumes
   import matplotlib.pyplot as plt
   
   with h5py.File('analysis_processed.h5', 'r') as f:
       volumes = f['cell_volume'][...]
       
   plt.hist(volumes, bins=50)
   plt.xlabel('Cell Volume (Ų)')
   plt.ylabel('Frequency')
   plt.title('Cell Volume Distribution')
   plt.show()
   
   print(f"Mean volume: {np.mean(volumes):.1f} Ų")
   print(f"Volume range: {np.min(volumes):.1f} - {np.max(volumes):.1f} Ų")

Advanced Workflows
~~~~~~~~~~~~~~~~~

**Fragment Shape Analysis**

.. code-block:: python

   # Analyze molecular shapes using inertia tensors
   def analyze_fragment_shapes(f):
       shape_data = []
       
       for i in range(len(f['refcode_list'])):
           n_frags = f['n_fragments'][i]
           if n_frags > 0:
               inertia = f['fragment_inertia_eigvals'][i].reshape(n_frags, 3)
               
               for j in range(n_frags):
                   I1, I2, I3 = sorted(inertia[j])
                   asphericity = I3 - 0.5*(I1 + I2)
                   acylindricity = I2 - I1
                   
                   shape_data.append({
                       'refcode': f['refcode_list'][i].decode(),
                       'fragment_id': j,
                       'asphericity': asphericity,
                       'acylindricity': acylindricity
                   })
       
       return pd.DataFrame(shape_data)
   
   # Usage
   with h5py.File('analysis_processed.h5', 'r') as f:
       shapes_df = analyze_fragment_shapes(f)
   
   # Plot shape distribution
   plt.scatter(shapes_df['asphericity'], shapes_df['acylindricity'], alpha=0.6)
   plt.xlabel('Asphericity')
   plt.ylabel('Acylindricity')
   plt.title('Molecular Shape Distribution')
   plt.show()

**Contact Network Analysis**

.. code-block:: python

   # Build hydrogen bonding networks
   import networkx as nx
   
   def build_hbond_network(f, structure_idx):
       """Build hydrogen bonding network for a single structure."""
       refcode = f['refcode_list'][structure_idx].decode()
       n_hbonds = f['inter_hb_n_hbonds'][structure_idx]
       
       if n_hbonds == 0:
           return None
           
       # Get H-bond data
       donors = f['inter_hb_central_atom'][structure_idx].astype(str)
       acceptors = f['inter_hb_contact_atom'][structure_idx].astype(str)
       distances = f['inter_hb_length'][structure_idx]
       
       # Build network
       G = nx.Graph()
       for i in range(n_hbonds):
           G.add_edge(donors[i], acceptors[i], distance=distances[i])
       
       return G
   
   # Example usage
   with h5py.File('analysis_processed.h5', 'r') as f:
       network = build_hbond_network(f, 0)
       if network:
           print(f"Network has {network.number_of_nodes()} atoms")
           print(f"Network has {network.number_of_edges()} H-bonds")

Configuration Templates
----------------------

Common Configuration Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**High-Throughput Screening**

.. code-block:: json

   {
     "extraction": {
       "data_directory": "./hts_analysis",
       "data_prefix": "screening_run",
       "filters": {
         "target_z_prime_values": [1],
         "crystal_type": ["homomolecular"],
         "molecule_weight_limit": 600.0
       },
       "extraction_batch_size": 128,
       "post_extraction_batch_size": 64
     }
   }

**Pharmaceutical Focus**

.. code-block:: json

   {
     "extraction": {
       "data_directory": "./pharma_analysis",
       "data_prefix": "drug_molecules",
       "filters": {
         "target_z_prime_values": [1, 2],
         "crystal_type": ["homomolecular"],
         "molecule_formal_charges": [0, 1, -1],
         "target_species": ["C", "H", "N", "O", "S", "F", "Cl"],
         "molecule_weight_limit": 800.0
       }
     }
   }

**Materials Science**

.. code-block:: json

   {
     "extraction": {
       "data_directory": "./materials_analysis",
       "data_prefix": "metal_organic",
       "filters": {
         "target_z_prime_values": [1, 2, 4],
         "crystal_type": ["organometallic", "homomolecular"],
         "target_species": ["C", "H", "N", "O", "Fe", "Cu", "Zn"],
         "exclude_disorder": true
       }
     }
   }

Utility Functions
----------------

Commonly Used Helper Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Data Loading Utilities**

.. code-block:: python

   def load_structure_summary(hdf5_file):
       """Load basic structure information into a pandas DataFrame."""
       with h5py.File(hdf5_file, 'r') as f:
           data = {
               'refcode': f['refcode_list'][...].astype(str),
               'space_group': [f['space_group'][i].decode() for i in range(len(f['refcode_list']))],
               'z_prime': f['z_prime'][...],
               'cell_volume': f['cell_volume'][...],
               'cell_density': f['cell_density'][...],
               'n_atoms': f['n_atoms'][...],
               'n_fragments': f['n_fragments'][...],
               'n_contacts': f['inter_cc_n_contacts'][...]
           }
       return pd.DataFrame(data)

**Filtering Utilities**

.. code-block:: python

   def filter_structures(df, criteria):
       """Apply multiple filtering criteria to structure DataFrame."""
       mask = pd.Series(True, index=df.index)
       
       if 'min_volume' in criteria:
           mask &= df['cell_volume'] >= criteria['min_volume']
       if 'max_volume' in criteria:
           mask &= df['cell_volume'] <= criteria['max_volume']
       if 'space_groups' in criteria:
           mask &= df['space_group'].isin(criteria['space_groups'])
       if 'z_prime_values' in criteria:
           mask &= df['z_prime'].isin(criteria['z_prime_values'])
           
       return df[mask]

**Visualization Utilities**

.. code-block:: python

   def plot_property_comparison(df, x_col, y_col, color_col=None):
       """Create scatter plot comparing two properties."""
       plt.figure(figsize=(10, 6))
       
       if color_col:
           scatter = plt.scatter(df[x_col], df[y_col], c=df[color_col], alpha=0.6)
           plt.colorbar(scatter, label=color_col)
       else:
           plt.scatter(df[x_col], df[y_col], alpha=0.6)
       
       plt.xlabel(x_col)
       plt.ylabel(y_col)
       plt.title(f'{y_col} vs {x_col}')
       plt.show()

Interactive Examples
-------------------

Jupyter Notebook Examples
~~~~~~~~~~~~~~~~~~~~~~~~~

**Example 1: Quick Dataset Exploration**

.. code-block:: python

   # Cell 1: Load and explore
   import h5py
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   
   # Load summary data
   df = load_structure_summary('my_analysis_processed.h5')
   df.head()

.. code-block:: python

   # Cell 2: Basic statistics
   df.describe()

.. code-block:: python

   # Cell 3: Visualizations
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   
   # Volume distribution
   axes[0,0].hist(df['cell_volume'], bins=30)
   axes[0,0].set_title('Cell Volume Distribution')
   
   # Density vs Volume
   axes[0,1].scatter(df['cell_volume'], df['cell_density'], alpha=0.6)
   axes[0,1].set_xlabel('Volume')
   axes[0,1].set_ylabel('Density')
   
   # Z' distribution
   df['z_prime'].value_counts().plot(kind='bar', ax=axes[1,0])
   axes[1,0].set_title("Z' Distribution")
   
   # Space group popularity
   df['space_group'].value_counts().head(10).plot(kind='barh', ax=axes[1,1])
   axes[1,1].set_title('Top Space Groups')
   
   plt.tight_layout()
   plt.show()

Command Line Examples
~~~~~~~~~~~~~~~~~~~~

**Quick Analysis Commands**

.. code-block:: bash

   # Run with default configuration
   python src/csa_main.py --config default.json
   
   # Run with custom batch size
   python src/csa_main.py --config analysis.json --batch-size 64
   
   # Run only specific stages
   python src/csa_main.py --config analysis.json --stages 4,5
   
   # Resume from checkpoint
   python src/csa_main.py --config analysis.json --resume

**Data Export Commands**

.. code-block:: bash

   # Export to CSV
   python scripts/export_data.py structures_processed.h5 --format csv --output analysis_data/
   
   # Export filtered subset
   python scripts/export_data.py structures_processed.h5 --filter "z_prime==1" --output subset/
   
   # Export specific properties
   python scripts/export_data.py structures_processed.h5 --properties cell_volume,cell_density

Troubleshooting Examples
-----------------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Memory Management**

.. code-block:: python

   # Problem: Out of memory when loading large datasets
   # Solution: Use chunked loading
   
   def load_data_chunked(hdf5_file, chunk_size=1000):
       """Load large datasets in chunks to manage memory."""
       results = []
       
       with h5py.File(hdf5_file, 'r') as f:
           n_structures = len(f['refcode_list'])
           
           for start in range(0, n_structures, chunk_size):
               end = min(start + chunk_size, n_structures)
               chunk_data = {
                   'refcode': f['refcode_list'][start:end].astype(str),
                   'cell_volume': f['cell_volume'][start:end]
               }
               results.append(pd.DataFrame(chunk_data))
       
       return pd.concat(results, ignore_index=True)

**Performance Optimization**

.. code-block:: python

   # Problem: Slow data access
   # Solution: Use indexing and caching
   
   def create_index_cache(hdf5_file):
       """Create an index for faster data access."""
       with h5py.File(hdf5_file, 'r') as f:
           index = {
               refcode: i for i, refcode in 
               enumerate(f['refcode_list'][...].astype(str))
           }
       return index
   
   # Usage
   index = create_index_cache('structures_processed.h5')
   target_idx = index['AABHTZ']

Example Downloads
----------------

All examples are available for download:

.. code-block:: bash

   # Download example files
   wget https://csa-examples.readthedocs.io/downloads/examples.zip
   unzip examples.zip
   cd csa_examples/
   
   # Run example scripts
   python basic_analysis_example.py
   python fragment_analysis_example.py
   python visualization_example.py

Browse Examples by Category
--------------------------

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Quick Start

   quick_analysis
   oneliners
   config_examples
   basic_plots

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Data Access

   reading_data
   data_filtering
   data_export
   data_conversion

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Analysis

   geometry_examples
   fragment_examples
   contact_examples
   statistics_examples

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Visualization

   distribution_plots
   correlation_plots
   structure_plots
   network_plots

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Integration

   pandas_examples
   sklearn_examples
   numpy_examples
   jupyter_examples

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Performance

   gpu_examples
   batch_examples
   memory_examples
   profiling_examples