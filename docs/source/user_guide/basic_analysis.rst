Basic Analysis
==============

This guide walks through fundamental analysis workflows using CSA, from running your first pipeline to extracting meaningful insights from crystal structure data. You'll learn to execute the five-stage pipeline, access results, and perform common analyses.

.. note::
   
   This guide assumes you have CSA installed and configured. See :doc:`../getting_started/installation` and :doc:`../getting_started/configuration` if you need setup help.

Complete Pipeline Workflow
---------------------------

Running the Five-Stage Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CSA pipeline transforms raw CSD queries into analysis-ready datasets through five sequential stages. Here's how to execute the complete workflow:

**Step 1: Create Configuration**

.. code-block:: json

    {
      "extraction": {
        "data_directory": "./basic_analysis_output",
        "data_prefix": "my_first_analysis",
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

**Step 2: Execute Pipeline**

.. code-block:: bash

    # Navigate to CSA directory
    cd /path/to/crystal-structure-analysis
    
    # Run complete pipeline
    python src/csa_main.py --config basic_analysis.json
    
    # Monitor progress
    tail -f basic_analysis_output/logs/csa_extraction.log

**Step 3: Verify Output**

After successful completion, you should see these files:

.. code-block:: text

    basic_analysis_output/
    ├── csv/
    │   ├── my_first_analysis_refcode_families.csv
    │   ├── my_first_analysis_clustered_families.csv
    │   └── my_first_analysis_unique_structures.csv
    ├── structures/
    │   ├── my_first_analysis_structures.h5
    │   └── my_first_analysis_structures_processed.h5
    └── logs/
        ├── csa_extraction.log
        └── performance_metrics.log

Understanding Pipeline Stages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each stage builds upon the previous, creating increasingly refined datasets:

**Stage 1: Family Extraction**

**Input**: CSD database query
**Output**: ``refcode_families.csv``
**Purpose**: Groups structures into chemical families

.. code-block:: python

    import pandas as pd
    
    # Load family data
    families = pd.read_csv('basic_analysis_output/csv/my_first_analysis_refcode_families.csv')
    print(f"Found {len(families)} structures across {families['family'].nunique()} families")
    
    # Examine family sizes
    family_sizes = families['family'].value_counts()
    print(f"Largest family: {family_sizes.iloc[0]} structures")
    print(f"Average family size: {family_sizes.mean():.1f}")

**Stage 2: Similarity Clustering**

**Input**: Refcode families
**Output**: ``clustered_families.csv``  
**Purpose**: Groups similar crystal packings within families

.. code-block:: python

    # Load clustering results
    clustered = pd.read_csv('basic_analysis_output/csv/my_first_analysis_clustered_families.csv')
    
    # Analyze clustering effectiveness
    cluster_sizes = clustered.groupby(['family', 'cluster']).size()
    print(f"Total clusters: {len(cluster_sizes)}")
    print(f"Average cluster size: {cluster_sizes.mean():.1f}")

**Stage 3: Representative Selection**

**Input**: Clustered families
**Output**: ``unique_structures.csv``
**Purpose**: Selects one representative per cluster

.. code-block:: python

    # Load selected structures
    unique = pd.read_csv('basic_analysis_output/csv/my_first_analysis_unique_structures.csv')
    print(f"Selected {len(unique)} representative structures")
    
    # Check selection criteria
    print(unique[['refcode', 'family', 'cluster', 'selection_score']].head())

**Stage 4: Structure Data Extraction**

**Input**: Representative structure list
**Output**: ``structures.h5`` (raw HDF5 data)
**Purpose**: Extracts atomic coordinates, bonds, and basic properties

.. code-block:: python

    import h5py
    
    # Examine raw data structure
    with h5py.File('basic_analysis_output/structures/my_first_analysis_structures.h5', 'r') as f:
        print("Raw data structure:")
        for group_name in f.keys():
            print(f"  /{group_name}")
            group = f[group_name]
            if hasattr(group, 'keys'):
                for dataset in list(group.keys())[:5]:  # Show first 5
                    print(f"    /{group_name}/{dataset}")

**Stage 5: Feature Engineering**

**Input**: Raw structural data
**Output**: ``structures_processed.h5`` (analysis-ready data)
**Purpose**: Computes advanced geometric and topological features

.. code-block:: python

    # Examine processed data
    with h5py.File('basic_analysis_output/structures/my_first_analysis_structures_processed.h5', 'r') as f:
        n_structures = len(f['refcode_list'])
        print(f"Processed {n_structures} structures")
        
        # Show available datasets
        print("Available datasets:")
        for dataset in sorted(f.keys()):
            shape = f[dataset].shape if hasattr(f[dataset], 'shape') else 'variable'
            print(f"  {dataset}: {shape}")

Accessing and Exploring Results
-------------------------------

Loading Processed Data
~~~~~~~~~~~~~~~~~~~~~~

The processed HDF5 file contains all computed features organized for efficient access:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import h5py
    from data_reader import RawDataReader

    # Open processed data file
    with h5py.File('basic_analysis_output/structures/my_first_analysis_structures_processed.h5', 'r') as f:
        reader = RawDataReader(f)
        
        # Load structure identifiers
        refcodes = f['refcode_list'][...].astype(str)
        n_structures = len(refcodes)
        
        print(f"Dataset contains {n_structures} structures")
        print(f"First 5 refcodes: {refcodes[:5]}")

Basic Data Exploration
~~~~~~~~~~~~~~~~~~~~~~

**Crystal Properties Overview**

.. code-block:: python

    def explore_crystal_properties(f):
        """Explore basic crystal-level properties."""
        
        # Load crystal data
        data = {
            'refcode': f['refcode_list'][...].astype(str),
            'space_group': [f['space_group'][i].decode() for i in range(len(f['refcode_list']))],
            'z_prime': f['z_prime'][...],
            'cell_volume': f['cell_volume'][...],
            'cell_density': f['cell_density'][...],
            'n_atoms': f['n_atoms'][...],
            'n_fragments': f['n_fragments'][...]
        }
        
        df = pd.DataFrame(data)
        
        # Basic statistics
        print("Crystal Properties Summary:")
        print(df.describe())
        
        # Space group distribution
        print(f"\nMost common space groups:")
        print(df['space_group'].value_counts().head(10))
        
        return df

    # Run exploration
    crystal_df = explore_crystal_properties(f)

**Fragment Analysis**

.. code-block:: python

    def analyze_fragments(f):
        """Analyze molecular fragment properties."""
        
        fragment_data = []
        
        for i in range(len(f['refcode_list'])):
            refcode = f['refcode_list'][i].decode()
            n_frags = f['n_fragments'][i]
            
            if n_frags > 0:
                # Fragment formulas
                formulas = f['fragment_formula'][i].astype(str)
                
                # Centers of mass
                com_flat = f['fragment_com_coords'][i]
                com_coords = com_flat.reshape(n_frags, 3)
                
                # Inertia eigenvalues (shape descriptors)
                inertia_flat = f['fragment_inertia_eigvals'][i]
                inertia_eigvals = inertia_flat.reshape(n_frags, 3)
                
                for j in range(n_frags):
                    fragment_data.append({
                        'refcode': refcode,
                        'fragment_id': j,
                        'formula': formulas[j],
                        'com_x': com_coords[j, 0],
                        'com_y': com_coords[j, 1], 
                        'com_z': com_coords[j, 2],
                        'inertia_1': inertia_eigvals[j, 0],
                        'inertia_2': inertia_eigvals[j, 1],
                        'inertia_3': inertia_eigvals[j, 2],
                        'asphericity': inertia_eigvals[j, 2] - 0.5*(inertia_eigvals[j, 0] + inertia_eigvals[j, 1])
                    })
        
        return pd.DataFrame(fragment_data)

    fragment_df = analyze_fragments(f)
    print(f"Analyzed {len(fragment_df)} fragments")
    print(f"Fragment formulas: {fragment_df['formula'].value_counts().head()}")

**Contact Analysis**

.. code-block:: python

    def analyze_contacts(f):
        """Analyze intermolecular contacts."""
        
        contact_data = []
        
        for i in range(len(f['refcode_list'])):
            refcode = f['refcode_list'][i].decode()
            n_contacts = f['inter_cc_n_contacts'][i]
            
            if n_contacts > 0:
                # Contact information
                central_atoms = f['inter_cc_central_atom'][i].astype(str)
                contact_atoms = f['inter_cc_contact_atom'][i].astype(str)
                lengths = f['inter_cc_length'][i]
                is_hbond = f['inter_cc_is_hbond'][i]
                
                for j in range(n_contacts):
                    contact_data.append({
                        'refcode': refcode,
                        'central_atom': central_atoms[j],
                        'contact_atom': contact_atoms[j],
                        'length': lengths[j],
                        'is_hbond': is_hbond[j]
                    })
        
        return pd.DataFrame(contact_data)

    contact_df = analyze_contacts(f)
    print(f"Analyzed {len(contact_df)} intermolecular contacts")
    print(f"Hydrogen bonds: {contact_df['is_hbond'].sum()} ({contact_df['is_hbond'].mean()*100:.1f}%)")

Common Analysis Patterns
------------------------

Statistical Analysis
~~~~~~~~~~~~~~~~~~~~

**Distribution Analysis**

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns

    def plot_property_distributions(crystal_df):
        """Plot distributions of key crystal properties."""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # Cell volume distribution
        axes[0].hist(crystal_df['cell_volume'], bins=50, alpha=0.7)
        axes[0].set_xlabel('Cell Volume (Ų)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Cell Volume Distribution')
        
        # Density distribution
        axes[1].hist(crystal_df['cell_density'], bins=50, alpha=0.7)
        axes[1].set_xlabel('Density (g/cm³)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Crystal Density Distribution')
        
        # Z' distribution
        z_counts = crystal_df['z_prime'].value_counts().sort_index()
        axes[2].bar(z_counts.index, z_counts.values)
        axes[2].set_xlabel("Z'")
        axes[2].set_ylabel('Count')
        axes[2].set_title("Z' Distribution")
        
        # Number of atoms
        axes[3].hist(crystal_df['n_atoms'], bins=50, alpha=0.7)
        axes[3].set_xlabel('Number of Atoms')
        axes[3].set_ylabel('Frequency')
        axes[3].set_title('Molecular Size Distribution')
        
        # Number of fragments
        axes[4].hist(crystal_df['n_fragments'], bins=max(crystal_df['n_fragments'])+1, alpha=0.7)
        axes[4].set_xlabel('Number of Fragments')
        axes[4].set_ylabel('Frequency')
        axes[4].set_title('Fragment Count Distribution')
        
        # Space group popularity
        top_sgs = crystal_df['space_group'].value_counts().head(10)
        axes[5].barh(range(len(top_sgs)), top_sgs.values)
        axes[5].set_yticks(range(len(top_sgs)))
        axes[5].set_yticklabels(top_sgs.index)
        axes[5].set_xlabel('Count')
        axes[5].set_title('Top 10 Space Groups')
        
        plt.tight_layout()
        plt.savefig('property_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

    plot_property_distributions(crystal_df)

**Correlation Analysis**

.. code-block:: python

    def analyze_correlations(crystal_df):
        """Analyze correlations between crystal properties."""
        
        # Select numeric columns
        numeric_cols = ['z_prime', 'cell_volume', 'cell_density', 'n_atoms', 'n_fragments']
        corr_data = crystal_df[numeric_cols]
        
        # Compute correlation matrix
        correlation_matrix = corr_data.corr()
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f')
        plt.title('Crystal Property Correlations')
        plt.tight_layout()
        plt.savefig('correlations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlation_matrix

    correlations = analyze_correlations(crystal_df)

Geometric Analysis
~~~~~~~~~~~~~~~~~~

**Molecular Shape Analysis**

.. code-block:: python

    def analyze_molecular_shapes(fragment_df):
        """Analyze molecular shape descriptors."""
        
        # Calculate shape descriptors from inertia moments
        fragment_df['acylindricity'] = fragment_df['inertia_2'] - fragment_df['inertia_1']
        fragment_df['relative_anisotropy'] = (2*fragment_df['asphericity']**2 + 0.75*fragment_df['acylindricity']**2) / (fragment_df['inertia_1']**2 + fragment_df['inertia_2']**2 + fragment_df['inertia_3']**2)
        
        # Shape classification
        def classify_shape(row):
            if row['asphericity'] < 0.1 and row['acylindricity'] < 0.1:
                return 'spherical'
            elif row['acylindricity'] < 0.1:
                return 'oblate'
            elif row['asphericity'] > 0.3:
                return 'prolate'
            else:
                return 'intermediate'
        
        fragment_df['shape_class'] = fragment_df.apply(classify_shape, axis=1)
        
        # Plot shape distribution
        shape_counts = fragment_df['shape_class'].value_counts()
        
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.pie(shape_counts.values, labels=shape_counts.index, autopct='%1.1f%%')
        plt.title('Molecular Shape Distribution')
        
        plt.subplot(1, 2, 2)
        plt.scatter(fragment_df['asphericity'], fragment_df['acylindricity'], 
                   c=fragment_df['shape_class'].astype('category').cat.codes, alpha=0.6)
        plt.xlabel('Asphericity')
        plt.ylabel('Acylindricity')
        plt.title('Shape Parameter Space')
        plt.colorbar(label='Shape Class')
        
        plt.tight_layout()
        plt.savefig('molecular_shapes.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fragment_df

    fragment_df = analyze_molecular_shapes(fragment_df)

**Contact Distance Analysis**

.. code-block:: python

    def analyze_contact_distances(contact_df):
        """Analyze distribution of intermolecular contact distances."""
        
        # Separate hydrogen bonds from other contacts
        hbonds = contact_df[contact_df['is_hbond'] == True]
        other_contacts = contact_df[contact_df['is_hbond'] == False]
        
        # Plot distance distributions
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(other_contacts['length'], bins=50, alpha=0.7, label='Other contacts', color='blue')
        plt.hist(hbonds['length'], bins=50, alpha=0.7, label='Hydrogen bonds', color='red')
        plt.xlabel('Contact Distance (Å)')
        plt.ylabel('Frequency')
        plt.title('Intermolecular Contact Distances')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        # Box plot by contact type
        contact_types = []
        distances = []
        
        for _, row in contact_df.iterrows():
            contact_type = f"{row['central_atom']}-{row['contact_atom']}"
            contact_types.append(contact_type)
            distances.append(row['length'])
        
        contact_type_df = pd.DataFrame({'type': contact_types, 'distance': distances})
        top_types = contact_type_df['type'].value_counts().head(10).index
        
        filtered_df = contact_type_df[contact_type_df['type'].isin(top_types)]
        sns.boxplot(data=filtered_df, x='type', y='distance')
        plt.xticks(rotation=45)
        plt.xlabel('Contact Type')
        plt.ylabel('Distance (Å)')
        plt.title('Distance by Contact Type')
        
        plt.tight_layout()
        plt.savefig('contact_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistics
        print(f"Average contact distance: {contact_df['length'].mean():.2f} Å")
        print(f"Average H-bond distance: {hbonds['length'].mean():.2f} Å")
        print(f"Average other contact distance: {other_contacts['length'].mean():.2f} Å")

    analyze_contact_distances(contact_df)

Advanced Analysis Workflows
---------------------------

Comparative Studies
~~~~~~~~~~~~~~~~~~~

**Space Group Comparison**

.. code-block:: python

    def compare_space_groups(crystal_df, top_n=5):
        """Compare properties across different space groups."""
        
        # Select most common space groups
        top_sgs = crystal_df['space_group'].value_counts().head(top_n).index
        sg_data = crystal_df[crystal_df['space_group'].isin(top_sgs)]
        
        # Compare key properties
        properties = ['cell_volume', 'cell_density', 'n_atoms']
        
        fig, axes = plt.subplots(1, len(properties), figsize=(15, 5))
        
        for i, prop in enumerate(properties):
            sns.boxplot(data=sg_data, x='space_group', y=prop, ax=axes[i])
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
            axes[i].set_title(f'{prop} by Space Group')
        
        plt.tight_layout()
        plt.savefig('space_group_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistical tests
        from scipy.stats import kruskal
        
        for prop in properties:
            groups = [sg_data[sg_data['space_group'] == sg][prop].values for sg in top_sgs]
            statistic, p_value = kruskal(*groups)
            print(f"{prop}: Kruskal-Wallis H = {statistic:.3f}, p = {p_value:.3e}")

    compare_space_groups(crystal_df)

**Z' Effect Analysis**

.. code-block:: python

    def analyze_zprime_effects(crystal_df, contact_df):
        """Analyze how Z' affects crystal properties and contacts."""
        
        # Group by Z'
        zprime_groups = crystal_df.groupby('z_prime')
        
        # Property comparisons
        zprime_stats = zprime_groups.agg({
            'cell_volume': ['mean', 'std'],
            'cell_density': ['mean', 'std'],
            'n_atoms': ['mean', 'std']
        }).round(3)
        
        print("Properties by Z':")
        print(zprime_stats)
        
        # Contact analysis by Z'
        contact_merged = contact_df.merge(crystal_df[['refcode', 'z_prime']], on='refcode')
        contact_stats = contact_merged.groupby('z_prime').agg({
            'length': ['mean', 'std', 'count'],
            'is_hbond': 'mean'
        }).round(3)
        
        print("\nContacts by Z':")
        print(contact_stats)
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Density vs Z'
        sns.boxplot(data=crystal_df, x='z_prime', y='cell_density', ax=axes[0,0])
        axes[0,0].set_title('Density vs Z\'')
        
        # Volume vs Z'
        sns.boxplot(data=crystal_df, x='z_prime', y='cell_volume', ax=axes[0,1])
        axes[0,1].set_title('Volume vs Z\'')
        
        # Contact distances vs Z'
        sns.boxplot(data=contact_merged, x='z_prime', y='length', ax=axes[1,0])
        axes[1,0].set_title('Contact Distances vs Z\'')
        
        # H-bond frequency vs Z'
        hbond_freq = contact_merged.groupby('z_prime')['is_hbond'].mean()
        axes[1,1].bar(hbond_freq.index, hbond_freq.values)
        axes[1,1].set_xlabel('Z\'')
        axes[1,1].set_ylabel('H-bond Frequency')
        axes[1,1].set_title('H-bond Frequency vs Z\'')
        
        plt.tight_layout()
        plt.savefig('zprime_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    analyze_zprime_effects(crystal_df, contact_df)

Exporting Results
-----------------

Data Export Formats
~~~~~~~~~~~~~~~~~~~

**CSV Export for External Analysis**

.. code-block:: python

    def export_analysis_results(crystal_df, fragment_df, contact_df, output_dir='./analysis_exports'):
        """Export analysis results in various formats."""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Crystal properties summary
        crystal_summary = crystal_df.groupby('space_group').agg({
            'cell_volume': ['mean', 'std', 'min', 'max'],
            'cell_density': ['mean', 'std', 'min', 'max'],
            'n_atoms': ['mean', 'std', 'min', 'max'],
            'n_fragments': ['mean', 'std', 'min', 'max']
        }).round(3)
        
        crystal_summary.to_csv(f'{output_dir}/crystal_properties_by_space_group.csv')
        
        # Fragment analysis summary
        fragment_summary = fragment_df.groupby('formula').agg({
            'asphericity': ['mean', 'std'],
            'acylindricity': ['mean', 'std'],
            'inertia_1': ['mean', 'std'],
            'inertia_2': ['mean', 'std'],
            'inertia_3': ['mean', 'std']
        }).round(3)
        
        fragment_summary.to_csv(f'{output_dir}/fragment_properties_by_formula.csv')
        
        # Contact statistics
        contact_summary = contact_df.groupby(['central_atom', 'contact_atom']).agg({
            'length': ['mean', 'std', 'min', 'max', 'count'],
            'is_hbond': 'mean'
        }).round(3)
        
        contact_summary.to_csv(f'{output_dir}/contact_statistics.csv')
        
        print(f"Analysis results exported to {output_dir}/")

    export_analysis_results(crystal_df, fragment_df, contact_df)

**HDF5 Subset Export**

.. code-block:: python

    def export_filtered_dataset(input_file, output_file, refcode_list):
        """Export a subset of structures to a new HDF5 file."""
        
        with h5py.File(input_file, 'r') as f_in:
            with h5py.File(output_file, 'w') as f_out:
                
                # Find indices of selected refcodes
                all_refcodes = f_in['refcode_list'][...].astype(str)
                indices = [i for i, ref in enumerate(all_refcodes) if ref in refcode_list]
                
                # Copy selected data
                f_out.create_dataset('refcode_list', 
                                   data=[all_refcodes[i] for i in indices],
                                   dtype=h5py.string_dtype('utf-8'))
                
                # Copy fixed-length datasets
                for dataset_name in ['z_prime', 'cell_volume', 'cell_density', 'n_atoms']:
                    if dataset_name in f_in:
                        data = f_in[dataset_name][indices]
                        f_out.create_dataset(dataset_name, data=data)
                
                # Copy variable-length datasets
                for dataset_name in ['atom_coords', 'atom_label', 'fragment_formula']:
                    if dataset_name in f_in:
                        data = [f_in[dataset_name][i] for i in indices]
                        f_out.create_dataset(dataset_name, data=data, dtype=f_in[dataset_name].dtype)
                
                print(f"Exported {len(indices)} structures to {output_file}")

    # Example: Export high-density structures
    high_density_refs = crystal_df[crystal_df['cell_density'] > 1.5]['refcode'].tolist()
    export_filtered_dataset(
        'basic_analysis_output/structures/my_first_analysis_structures_processed.h5',
        'high_density_structures.h5',
        high_density_refs
    )

Performance Monitoring
----------------------

Tracking Analysis Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import time
    import psutil
    import os

    def monitor_analysis_performance(analysis_function, *args, **kwargs):
        """Monitor memory and time usage of analysis functions."""
        
        # Initial memory measurement
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time the analysis
        start_time = time.time()
        result = analysis_function(*args, **kwargs)
        end_time = time.time()
        
        # Final memory measurement
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = process.memory_info().peak_wss / 1024 / 1024 if hasattr(process.memory_info(), 'peak_wss') else final_memory
        
        # Report performance
        print(f"Analysis Performance:")
        print(f"  Execution time: {end_time - start_time:.2f} seconds")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Memory increase: {final_memory - initial_memory:.1f} MB")
        
        return result

    # Example usage
    fragment_df = monitor_analysis_performance(analyze_fragments, f)

Next Steps
----------

With basic analysis mastery:

1. **Explore specialized tutorials**: Learn domain-specific analysis techniques
2. **Try advanced workflows**: Experiment with machine learning and statistical modeling
3. **Develop custom analyses**: Create your own analysis functions and workflows
4. **Share and collaborate**: Export results for publication and collaboration
5. **Scale up**: Apply techniques to larger datasets and production workflows

See Also
--------

:doc:`../tutorials/index` : Step-by-step tutorials for specific analyses
:doc:`../examples/index` : Ready-to-run analysis examples
:doc:`data_model` : Understanding CSA's data organization
:doc:`../technical_details/performance` : Performance optimization techniques