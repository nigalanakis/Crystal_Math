Basic Analysis
==============

This guide walks through fundamental analysis workflows using CSA, from running your first pipeline to extracting meaningful insights from crystal structure data. You'll learn to execute the five-stage pipeline, access results, and perform common analyses.

.. note::
   
   This guide assumes you have CSA installed and configured. See :doc:`../getting_started/installation` and :doc:`../getting_started/configuration` if you need setup help.

Getting Started: Your First Analysis
------------------------------------

Complete Pipeline Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Understanding Pipeline Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each stage creates specific outputs that feed into subsequent analyses:

**Stage 1: Family Extraction** → ``refcode_families.csv``
   Groups structures into chemical families

**Stage 2: Similarity Clustering** → ``clustered_families.csv``
   Groups similar crystal packings within families

**Stage 3: Representative Selection** → ``unique_structures.csv``
   Selects one representative per cluster

**Stage 4: Structure Data Extraction** → ``structures.h5``
   Raw structural data with coordinates and properties

**Stage 5: Feature Engineering** → ``structures_processed.h5``
   Analysis-ready data with computed features

Quick validation of your results:

.. code-block:: python

    import pandas as pd
    import h5py
    
    # Check family extraction results
    families = pd.read_csv('basic_analysis_output/csv/my_first_analysis_refcode_families.csv')
    print(f"Found {len(families)} structures across {families['family'].nunique()} families")
    
    # Check processed data
    with h5py.File('basic_analysis_output/structures/my_first_analysis_structures_processed.h5', 'r') as f:
        n_structures = len(f['refcode_list'])
        print(f"Successfully processed {n_structures} representative structures")

Accessing Your Data
-------------------

Loading and Exploring Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CSA stores results in HDF5 format for efficient access. Here's how to load and explore your data:

**Basic Data Loading**

.. code-block:: python

    import numpy as np
    import pandas as pd
    import h5py

    def load_crystal_data(hdf5_file):
        """Load basic crystal properties into a pandas DataFrame."""
        
        with h5py.File(hdf5_file, 'r') as f:
            data = {
                'refcode': f['refcode_list'][...].astype(str),
                'space_group': [f['space_group'][i].decode() for i in range(len(f['refcode_list']))],
                'z_prime': f['z_prime'][...],
                'cell_volume': f['cell_volume'][...],
                'cell_density': f['cell_density'][...],
                'n_atoms': f['n_atoms'][...],
                'n_fragments': f['n_fragments'][...],
                'temperature': f['temperature'][...]
            }
        
        return pd.DataFrame(data)

    # Load your data
    crystal_df = load_crystal_data('basic_analysis_output/structures/my_first_analysis_structures_processed.h5')
    print(crystal_df.head())

**Data Summary and Statistics**

.. code-block:: python

    def summarize_dataset(crystal_df):
        """Generate a comprehensive dataset summary."""
        
        print("="*50)
        print("DATASET SUMMARY")
        print("="*50)
        
        print(f"Total structures: {len(crystal_df)}")
        print(f"Unique space groups: {crystal_df['space_group'].nunique()}")
        print(f"Temperature range: {crystal_df['temperature'].min():.0f} - {crystal_df['temperature'].max():.0f} K")
        
        print("\nMolecular size distribution:")
        print(crystal_df['n_atoms'].describe())
        
        print("\nZ' distribution:")
        print(crystal_df['z_prime'].value_counts().sort_index())
        
        print("\nTop 10 space groups:")
        print(crystal_df['space_group'].value_counts().head(10))
        
        return crystal_df.describe()

    summary_stats = summarize_dataset(crystal_df)
    print(summary_stats)

Essential Analysis Workflows
----------------------------

Crystal Property Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

**Distribution Analysis**

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns

    def plot_crystal_properties(crystal_df):
        """Plot key crystal property distributions."""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # Cell volume distribution
        axes[0].hist(crystal_df['cell_volume'], bins=50, alpha=0.7, color='skyblue')
        axes[0].set_xlabel('Cell Volume (ų)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Cell Volume Distribution')
        
        # Density distribution
        axes[1].hist(crystal_df['cell_density'], bins=50, alpha=0.7, color='lightgreen')
        axes[1].set_xlabel('Density (g/cm³)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Crystal Density Distribution')
        
        # Z' distribution
        z_counts = crystal_df['z_prime'].value_counts().sort_index()
        axes[2].bar(z_counts.index, z_counts.values, color='coral')
        axes[2].set_xlabel("Z'")
        axes[2].set_ylabel('Count')
        axes[2].set_title("Z' Distribution")
        
        # Molecular size
        axes[3].hist(crystal_df['n_atoms'], bins=50, alpha=0.7, color='gold')
        axes[3].set_xlabel('Number of Atoms')
        axes[3].set_ylabel('Frequency')
        axes[3].set_title('Molecular Size Distribution')
        
        # Fragment count
        axes[4].hist(crystal_df['n_fragments'], bins=max(crystal_df['n_fragments'])+1, 
                    alpha=0.7, color='mediumpurple')
        axes[4].set_xlabel('Number of Fragments')
        axes[4].set_ylabel('Frequency')
        axes[4].set_title('Fragment Count Distribution')
        
        # Temperature distribution
        axes[5].hist(crystal_df['temperature'], bins=50, alpha=0.7, color='lightcoral')
        axes[5].set_xlabel('Temperature (K)')
        axes[5].set_ylabel('Frequency')
        axes[5].set_title('Measurement Temperature')
        
        plt.tight_layout()
        plt.savefig('crystal_property_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

    plot_crystal_properties(crystal_df)

**Correlation Analysis**

.. code-block:: python

    def analyze_property_correlations(crystal_df):
        """Analyze correlations between crystal properties."""
        
        # Select numeric properties for correlation analysis
        numeric_cols = ['z_prime', 'cell_volume', 'cell_density', 'n_atoms', 
                       'n_fragments', 'temperature']
        
        corr_data = crystal_df[numeric_cols]
        correlation_matrix = corr_data.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
        plt.title('Crystal Property Correlations')
        plt.tight_layout()
        plt.savefig('property_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Identify strong correlations
        print("Strong correlations (|r| > 0.5):")
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    prop1 = correlation_matrix.columns[i]
                    prop2 = correlation_matrix.columns[j]
                    print(f"{prop1} vs {prop2}: r = {corr_val:.3f}")
        
        return correlation_matrix

    correlations = analyze_property_correlations(crystal_df)

Fragment Analysis
~~~~~~~~~~~~~~~~

**Fragment Properties and Shapes**

.. code-block:: python

    def analyze_molecular_fragments(hdf5_file):
        """Analyze molecular fragment properties and shapes."""
        
        fragment_data = []
        
        with h5py.File(hdf5_file, 'r') as f:
            n_structures = len(f['refcode_list'])
            
            for i in range(n_structures):
                refcode = f['refcode_list'][i].decode()
                n_frags = f['n_fragments'][i]
                
                if n_frags > 0:
                    # Fragment formulas
                    formulas = f['fragment_formula'][i].astype(str)
                    
                    # Centers of mass coordinates
                    com_flat = f['fragment_com_coords'][i]
                    com_coords = com_flat.reshape(n_frags, 3)
                    
                    # Inertia eigenvalues for shape analysis
                    inertia_flat = f['fragment_inertia_eigvals'][i]
                    inertia_eigvals = inertia_flat.reshape(n_frags, 3)
                    
                    for j in range(n_frags):
                        # Calculate shape descriptors
                        asphericity = inertia_eigvals[j, 2] - 0.5*(inertia_eigvals[j, 0] + inertia_eigvals[j, 1])
                        acylindricity = inertia_eigvals[j, 1] - inertia_eigvals[j, 0]
                        
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
                            'asphericity': asphericity,
                            'acylindricity': acylindricity
                        })
        
        return pd.DataFrame(fragment_data)

    # Analyze fragments
    fragment_df = analyze_molecular_fragments('basic_analysis_output/structures/my_first_analysis_structures_processed.h5')
    print(f"Analyzed {len(fragment_df)} fragments")
    print("\nMost common fragment formulas:")
    print(fragment_df['formula'].value_counts().head(10))

**Fragment Shape Classification**

.. code-block:: python

    def classify_fragment_shapes(fragment_df):
        """Classify molecular fragments by shape."""
        
        def classify_shape(row):
            """Classify shape based on asphericity and acylindricity."""
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
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        shape_counts = fragment_df['shape_class'].value_counts()
        plt.pie(shape_counts.values, labels=shape_counts.index, autopct='%1.1f%%')
        plt.title('Fragment Shape Distribution')
        
        plt.subplot(1, 2, 2)
        colors = {'spherical': 'blue', 'oblate': 'green', 'prolate': 'red', 'intermediate': 'orange'}
        for shape in fragment_df['shape_class'].unique():
            data = fragment_df[fragment_df['shape_class'] == shape]
            plt.scatter(data['asphericity'], data['acylindricity'], 
                       c=colors[shape], label=shape, alpha=0.6)
        
        plt.xlabel('Asphericity')
        plt.ylabel('Acylindricity')
        plt.title('Shape Parameter Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fragment_shapes.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Shape classification results:")
        print(fragment_df['shape_class'].value_counts())
        
        return fragment_df

    fragment_df = classify_fragment_shapes(fragment_df)

Intermolecular Contact Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Contact Detection and Classification**

.. code-block:: python

    def analyze_intermolecular_contacts(hdf5_file):
        """Analyze intermolecular contacts and hydrogen bonds."""
        
        contact_data = []
        
        with h5py.File(hdf5_file, 'r') as f:
            n_structures = len(f['refcode_list'])
            
            for i in range(n_structures):
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
                            'is_hbond': bool(is_hbond[j])
                        })
        
        return pd.DataFrame(contact_data)

    # Analyze contacts
    contact_df = analyze_intermolecular_contacts('basic_analysis_output/structures/my_first_analysis_structures_processed.h5')
    print(f"Found {len(contact_df)} intermolecular contacts")
    print(f"Hydrogen bonds: {contact_df['is_hbond'].sum()} ({contact_df['is_hbond'].mean()*100:.1f}%)")

**Contact Distance Analysis**

.. code-block:: python

    def plot_contact_analysis(contact_df):
        """Plot contact distance distributions and types."""
        
        # Separate hydrogen bonds from other contacts
        hbonds = contact_df[contact_df['is_hbond'] == True]
        other_contacts = contact_df[contact_df['is_hbond'] == False]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Distance distributions
        axes[0,0].hist(other_contacts['length'], bins=50, alpha=0.7, 
                      label='Other contacts', color='blue', density=True)
        axes[0,0].hist(hbonds['length'], bins=50, alpha=0.7, 
                      label='Hydrogen bonds', color='red', density=True)
        axes[0,0].set_xlabel('Contact Distance (Å)')
        axes[0,0].set_ylabel('Density')
        axes[0,0].set_title('Contact Distance Distributions')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Contact type frequencies
        contact_types = []
        for _, row in contact_df.iterrows():
            contact_type = f"{row['central_atom']}-{row['contact_atom']}"
            contact_types.append(contact_type)
        
        contact_type_df = pd.DataFrame({'type': contact_types, 'distance': contact_df['length']})
        top_types = contact_type_df['type'].value_counts().head(10)
        
        axes[0,1].barh(range(len(top_types)), top_types.values)
        axes[0,1].set_yticks(range(len(top_types)))
        axes[0,1].set_yticklabels(top_types.index)
        axes[0,1].set_xlabel('Count')
        axes[0,1].set_title('Most Common Contact Types')
        
        # Box plot of distances by contact type
        top_types_list = top_types.index[:8].tolist()
        filtered_df = contact_type_df[contact_type_df['type'].isin(top_types_list)]
        
        if len(filtered_df) > 0:
            sns.boxplot(data=filtered_df, x='type', y='distance', ax=axes[1,0])
            axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45)
            axes[1,0].set_title('Distance Distributions by Contact Type')
        
        # H-bond vs other contact statistics
        stats_data = {
            'Contact Type': ['Hydrogen Bonds', 'Other Contacts'],
            'Count': [len(hbonds), len(other_contacts)],
            'Mean Distance': [hbonds['length'].mean() if len(hbonds) > 0 else 0, 
                            other_contacts['length'].mean()],
            'Std Distance': [hbonds['length'].std() if len(hbonds) > 0 else 0, 
                           other_contacts['length'].std()]
        }
        
        stats_df = pd.DataFrame(stats_data)
        axes[1,1].axis('tight')
        axes[1,1].axis('off')
        table = axes[1,1].table(cellText=stats_df.round(3).values,
                               colLabels=stats_df.columns,
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        axes[1,1].set_title('Contact Statistics Summary')
        
        plt.tight_layout()
        plt.savefig('contact_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\nContact Analysis Summary:")
        print(f"Total contacts: {len(contact_df)}")
        print(f"Hydrogen bonds: {len(hbonds)} ({len(hbonds)/len(contact_df)*100:.1f}%)")
        print(f"Average contact distance: {contact_df['length'].mean():.2f} ± {contact_df['length'].std():.2f} Å")
        
        if len(hbonds) > 0:
            print(f"Average H-bond distance: {hbonds['length'].mean():.2f} ± {hbonds['length'].std():.2f} Å")
        if len(other_contacts) > 0:
            print(f"Average other contact distance: {other_contacts['length'].mean():.2f} ± {other_contacts['length'].std():.2f} Å")

    plot_contact_analysis(contact_df)

Advanced Analysis Examples
--------------------------

Comparative Studies
~~~~~~~~~~~~~~~~~~

**Space Group Effects**

.. code-block:: python

    def compare_space_groups(crystal_df, contact_df, top_n=5):
        """Compare crystal properties across different space groups."""
        
        # Focus on most common space groups
        top_sgs = crystal_df['space_group'].value_counts().head(top_n).index
        sg_data = crystal_df[crystal_df['space_group'].isin(top_sgs)]
        
        # Statistical comparison
        print("Space Group Comparison:")
        print("="*50)
        
        for sg in top_sgs:
            sg_subset = sg_data[sg_data['space_group'] == sg]
            print(f"\n{sg} (n={len(sg_subset)}):")
            print(f"  Volume: {sg_subset['cell_volume'].mean():.1f} ± {sg_subset['cell_volume'].std():.1f} ų")
            print(f"  Density: {sg_subset['cell_density'].mean():.2f} ± {sg_subset['cell_density'].std():.2f} g/cm³")
            print(f"  Z': {sg_subset['z_prime'].mode()[0] if len(sg_subset['z_prime'].mode()) > 0 else 'N/A'} (mode)")
        
        # Visualize comparisons
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Volume comparison
        sns.boxplot(data=sg_data, x='space_group', y='cell_volume', ax=axes[0])
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
        axes[0].set_title('Cell Volume by Space Group')
        
        # Density comparison
        sns.boxplot(data=sg_data, x='space_group', y='cell_density', ax=axes[1])
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
        axes[1].set_title('Density by Space Group')
        
        # Z' distribution
        zprime_counts = sg_data.groupby(['space_group', 'z_prime']).size().unstack(fill_value=0)
        zprime_counts.plot(kind='bar', ax=axes[2], stacked=True)
        axes[2].set_title("Z' Distribution by Space Group")
        axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45)
        axes[2].legend(title="Z'")
        
        plt.tight_layout()
        plt.savefig('space_group_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    compare_space_groups(crystal_df, contact_df)

**Z' Effects Analysis**

.. code-block:: python

    def analyze_zprime_effects(crystal_df, contact_df):
        """Analyze how Z' affects crystal properties and intermolecular contacts."""
        
        # Merge contact data with crystal data
        contact_with_zprime = contact_df.merge(
            crystal_df[['refcode', 'z_prime']], on='refcode'
        )
        
        print("Z' Effects Analysis:")
        print("="*50)
        
        # Crystal property effects
        zprime_stats = crystal_df.groupby('z_prime').agg({
            'cell_volume': ['mean', 'std', 'count'],
            'cell_density': ['mean', 'std'],
            'n_atoms': ['mean', 'std']
        }).round(2)
        
        print("\nCrystal Properties by Z':")
        print(zprime_stats)
        
        # Contact effects
        contact_stats = contact_with_zprime.groupby('z_prime').agg({
            'length': ['mean', 'std', 'count'],
            'is_hbond': 'mean'
        }).round(3)
        
        print("\nContact Properties by Z':")
        print(contact_stats)
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Density vs Z'
        sns.boxplot(data=crystal_df, x='z_prime', y='cell_density', ax=axes[0,0])
        axes[0,0].set_title("Density vs Z'")
        
        # Volume vs Z'
        sns.boxplot(data=crystal_df, x='z_prime', y='cell_volume', ax=axes[0,1])
        axes[0,1].set_title("Volume vs Z'")
        
        # Contact distances vs Z'
        sns.boxplot(data=contact_with_zprime, x='z_prime', y='length', ax=axes[1,0])
        axes[1,0].set_title("Contact Distances vs Z'")
        
        # H-bond frequency vs Z'
        hbond_freq = contact_with_zprime.groupby('z_prime')['is_hbond'].mean()
        axes[1,1].bar(hbond_freq.index, hbond_freq.values)
        axes[1,1].set_xlabel("Z'")
        axes[1,1].set_ylabel('H-bond Frequency')
        axes[1,1].set_title("H-bond Frequency vs Z'")
        
        plt.tight_layout()
        plt.savefig('zprime_effects.png', dpi=300, bbox_inches='tight')
        plt.show()

    analyze_zprime_effects(crystal_df, contact_df)

Data Export and Sharing
-----------------------

Export Results
~~~~~~~~~~~~~

**CSV Export for External Analysis**

.. code-block:: python

    def export_analysis_results(crystal_df, fragment_df, contact_df, output_dir='analysis_exports'):
        """Export analysis results for external use."""
        
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
        if len(fragment_df) > 0:
            fragment_summary = fragment_df.groupby('formula').agg({
                'asphericity': ['mean', 'std'],
                'acylindricity': ['mean', 'std'],
                'inertia_1': ['mean', 'std'],
                'inertia_2': ['mean', 'std'],
                'inertia_3': ['mean', 'std']
            }).round(3)
            
            fragment_summary.to_csv(f'{output_dir}/fragment_properties_by_formula.csv')
        
        # Contact statistics
        if len(contact_df) > 0:
            contact_summary = contact_df.groupby(['central_atom', 'contact_atom']).agg({
                'length': ['mean', 'std', 'min', 'max', 'count'],
                'is_hbond': 'mean'
            }).round(3)
            
            contact_summary.to_csv(f'{output_dir}/contact_statistics.csv')
        
        # Export raw data
        crystal_df.to_csv(f'{output_dir}/crystal_data.csv', index=False)
        if len(fragment_df) > 0:
            fragment_df.to_csv(f'{output_dir}/fragment_data.csv', index=False)
        if len(contact_df) > 0:
            contact_df.to_csv(f'{output_dir}/contact_data.csv', index=False)
        
        print(f"Analysis results exported to {output_dir}/")
        print("Files created:")
        for file in os.listdir(output_dir):
            print(f"  - {file}")

    export_analysis_results(crystal_df, fragment_df, contact_df)

**Create Analysis Report**

.. code-block:: python

    def generate_analysis_report(crystal_df, fragment_df, contact_df, output_file='analysis_report.txt'):
        """Generate a comprehensive analysis report."""
        
        with open(output_file, 'w') as f:
            f.write("CSA Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Dataset overview
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total structures analyzed: {len(crystal_df)}\n")
            f.write(f"Unique space groups: {crystal_df['space_group'].nunique()}\n")
            f.write(f"Temperature range: {crystal_df['temperature'].min():.0f} - {crystal_df['temperature'].max():.0f} K\n")
            f.write(f"Total molecular fragments: {len(fragment_df)}\n")
            f.write(f"Total intermolecular contacts: {len(contact_df)}\n\n")
            
            # Crystal properties
            f.write("CRYSTAL PROPERTIES\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average cell volume: {crystal_df['cell_volume'].mean():.1f} ± {crystal_df['cell_volume'].std():.1f} ų\n")
            f.write(f"Average density: {crystal_df['cell_density'].mean():.2f} ± {crystal_df['cell_density'].std():.2f} g/cm³\n")
            f.write(f"Average molecular size: {crystal_df['n_atoms'].mean():.1f} ± {crystal_df['n_atoms'].std():.1f} atoms\n\n")
            
            # Most common space groups
            f.write("TOP SPACE GROUPS\n")
            f.write("-" * 20 + "\n")
            top_sgs = crystal_df['space_group'].value_counts().head(10)
            for sg, count in top_sgs.items():
                f.write(f"{sg}: {count} structures ({count/len(crystal_df)*100:.1f}%)\n")
            f.write("\n")
            
            # Contact analysis
            if len(contact_df) > 0:
                hbonds = contact_df[contact_df['is_hbond'] == True]
                f.write("INTERMOLECULAR CONTACTS\n")
                f.write("-" * 25 + "\n")
                f.write(f"Total contacts: {len(contact_df)}\n")
                f.write(f"Hydrogen bonds: {len(hbonds)} ({len(hbonds)/len(contact_df)*100:.1f}%)\n")
                f.write(f"Average contact distance: {contact_df['length'].mean():.2f} ± {contact_df['length'].std():.2f} Å\n")
                if len(hbonds) > 0:
                    f.write(f"Average H-bond distance: {hbonds['length'].mean():.2f} ± {hbonds['length'].std():.2f} Å\n")
                f.write("\n")
            
            # Fragment analysis
            if len(fragment_df) > 0:
                f.write("FRAGMENT ANALYSIS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Average asphericity: {fragment_df['asphericity'].mean():.3f} ± {fragment_df['asphericity'].std():.3f}\n")
                f.write(f"Average acylindricity: {fragment_df['acylindricity'].mean():.3f} ± {fragment_df['acylindricity'].std():.3f}\n")
                
                if 'shape_class' in fragment_df.columns:
                    f.write("\nShape distribution:\n")
                    for shape, count in fragment_df['shape_class'].value_counts().items():
                        f.write(f"  {shape}: {count} ({count/len(fragment_df)*100:.1f}%)\n")
        
        print(f"Analysis report saved to {output_file}")

    generate_analysis_report(crystal_df, fragment_df, contact_df)

Troubleshooting and Optimization
--------------------------------

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

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
        
        # Report performance
        print(f"Analysis Performance:")
        print(f"  Execution time: {end_time - start_time:.2f} seconds")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Memory increase: {final_memory - initial_memory:.1f} MB")
        
        return result

    # Example usage
    crystal_df = monitor_analysis_performance(load_crystal_data, 
                 'basic_analysis_output/structures/my_first_analysis_structures_processed.h5')

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Data Access Problems**

.. code-block:: python

    def validate_data_file(hdf5_file):
        """Check if HDF5 file is valid and contains expected data."""
        
        try:
            with h5py.File(hdf5_file, 'r') as f:
                # Check required datasets
                required_datasets = ['refcode_list', 'space_group', 'cell_volume', 'cell_density']
                missing_datasets = [ds for ds in required_datasets if ds not in f]
                
                if missing_datasets:
                    print(f"ERROR: Missing datasets: {missing_datasets}")
                    return False
                
                n_structures = len(f['refcode_list'])
                print(f"✓ File is valid with {n_structures} structures")
                
                # Check for common optional datasets
                optional_datasets = ['n_fragments', 'inter_cc_n_contacts', 'fragment_formula']
                available_optional = [ds for ds in optional_datasets if ds in f]
                print(f"✓ Available optional datasets: {available_optional}")
                
                return True
                
        except Exception as e:
            print(f"ERROR reading file: {e}")
            return False

    # Validate your data file
    validate_data_file('basic_analysis_output/structures/my_first_analysis_structures_processed.h5')

**Memory Issues with Large Datasets**

.. code-block:: python

    def load_data_efficiently(hdf5_file, chunk_size=1000):
        """Load large datasets efficiently using chunking."""
        
        results = []
        
        with h5py.File(hdf5_file, 'r') as f:
            n_structures = len(f['refcode_list'])
            
            for start in range(0, n_structures, chunk_size):
                end = min(start + chunk_size, n_structures)
                
                chunk_data = {
                    'refcode': f['refcode_list'][start:end].astype(str),
                    'cell_volume': f['cell_volume'][start:end],
                    'cell_density': f['cell_density'][start:end]
                }
                
                results.append(pd.DataFrame(chunk_data))
        
        return pd.concat(results, ignore_index=True)

    # Use for very large datasets
    # crystal_df = load_data_efficiently('your_large_dataset.h5')

Next Steps
----------

With basic analysis mastery, you're ready to:

1. **Explore Domain-Specific Tutorials**: Learn analysis techniques for your research area
   - :doc:`../tutorials/pharmaceutical_analysis` for drug development
   - :doc:`../tutorials/materials_science` for materials research
   - :doc:`../tutorials/organic_chemistry` for synthetic chemistry

2. **Try Advanced Workflows**: Experiment with sophisticated analysis techniques
   - :doc:`advanced_features` for complex analysis workflows
   - :doc:`custom_workflows` for specialized research questions

3. **Scale Your Analysis**: Apply techniques to larger datasets
   - :doc:`batch_processing` for high-throughput analysis
   - :doc:`../technical_details/performance` for optimization strategies

4. **Share and Collaborate**: Export results for publication and collaboration
   - :doc:`data_export` for sharing data and results
   - :doc:`../examples/index` for ready-to-run analysis scripts

5. **Develop Custom Analysis**: Create your own analysis functions
   - :doc:`../api_reference/index` for programmatic access
   - :doc:`../technical_details/architecture` for extending CSA

See Also
--------

:doc:`../tutorials/index` : Step-by-step tutorials for specific analyses
:doc:`../examples/index` : Ready-to-run analysis examples
:doc:`data_model` : Understanding CSA's data organization
:doc:`configuration` : Advanced configuration strategies
:doc:`../technical_details/performance` : Performance optimization techniques