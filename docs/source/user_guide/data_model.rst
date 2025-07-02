Data Model
==========

CSA organizes crystal structure data in a hierarchical, efficient format optimized for computational analysis. This guide explains how CSA represents molecular crystals, stores variable-length data, and enables fast access to structural information.

.. note::
   
   Understanding the data model is essential for writing analysis scripts and accessing CSA results programmatically.

Overview
--------

CSA uses HDF5 (Hierarchical Data Format) to store crystal structure data in a scientifically robust, cross-platform format. The data model is designed to handle the inherent complexity of crystallographic data while enabling efficient computation on modern hardware.

**Key Design Principles**:

- **Variable-Length Storage**: Crystals have different numbers of atoms, bonds, and contacts
- **Batch Processing**: Data organized for efficient GPU tensor operations  
- **Type Safety**: Strict data types prevent computational errors
- **Metadata Rich**: Complete provenance and experimental information
- **Compression**: Efficient storage with lossless compression

Data Hierarchy
--------------

CSA organizes data at four hierarchical levels:

.. code-block:: text

    Dataset Level
    ├── Crystal Level (per structure)
    │   ├── Molecule Level (per molecule in asymmetric unit)
    │   │   ├── Fragment Level (per rigid fragment)
    │   │   └── Atom Level (per atom)
    │   └── Interaction Level (intermolecular contacts)
    └── Metadata Level (experimental and computational details)

Dataset Level
~~~~~~~~~~~~~

The top level contains dataset-wide information and structure lists:

.. code-block:: python

    # Dataset structure overview
    with h5py.File('structures_processed.h5', 'r') as f:
        print("Dataset-level information:")
        print(f"Total structures: {len(f['refcode_list'])}")
        print(f"Processing date: {f.attrs.get('creation_date', 'N/A')}")
        print(f"CSA version: {f.attrs.get('csa_version', 'N/A')}")

**Key Datasets**:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Dataset Name
     - Data Type
     - Description
   * - ``refcode_list``
     - ``string``
     - CSD reference codes for all structures
   * - ``processing_status``
     - ``int8``
     - Success/failure flags for each structure
   * - ``computation_time``
     - ``float32``
     - Processing time per structure (seconds)

Crystal Level
~~~~~~~~~~~~~

Each crystal structure has associated properties describing the unit cell, space group, and overall characteristics:

.. code-block:: python

    # Crystal-level data access
    def get_crystal_properties(f, structure_index):
        """Extract crystal-level properties for a specific structure."""
        
        properties = {
            'refcode': f['refcode_list'][structure_index].decode(),
            'space_group': f['space_group'][structure_index].decode(),
            'z_prime': f['z_prime'][structure_index],
            'cell_volume': f['cell_volume'][structure_index],
            'cell_density': f['cell_density'][structure_index],
            'temperature': f['temperature'][structure_index],
            'pressure': f['pressure'][structure_index]
        }
        
        return properties

**Crystal Property Categories**:

**Unit Cell Parameters**:
- ``cell_a``, ``cell_b``, ``cell_c``: Lattice parameters (Å)
- ``cell_alpha``, ``cell_beta``, ``cell_gamma``: Lattice angles (degrees)
- ``cell_volume``: Unit cell volume (ų)

**Crystallographic Properties**:
- ``space_group``: Space group symbol (e.g., "P21/c")
- ``space_group_number``: International space group number
- ``z_prime``: Number of independent molecules in asymmetric unit
- ``crystal_system``: Crystal system classification

**Physical Properties**:
- ``cell_density``: Calculated density (g/cm³)
- ``packing_coefficient``: Fraction of space occupied by molecules
- ``temperature``: Measurement temperature (K)
- ``pressure``: Measurement pressure (typically 1 atm)

**Experimental Details**:
- ``resolution``: Diffraction resolution (Å)
- ``r_factor``: Crystallographic R-factor
- ``data_completeness``: Fraction of measured reflections
- ``radiation_type``: X-ray source (e.g., "Mo Kα")

Molecule Level
~~~~~~~~~~~~~~

Within each crystal, CSA tracks individual molecules in the asymmetric unit:

.. code-block:: python

    # Molecule-level data access
    def get_molecule_data(f, structure_index):
        """Extract molecular information for a structure."""
        
        n_molecules = f['n_molecules'][structure_index]
        
        molecules = []
        for mol_idx in range(n_molecules):
            # Molecular formula and properties
            formula = f['molecule_formula'][structure_index][mol_idx].decode()
            weight = f['molecule_weight'][structure_index][mol_idx]
            charge = f['molecule_formal_charge'][structure_index][mol_idx]
            
            molecules.append({
                'formula': formula,
                'molecular_weight': weight,
                'formal_charge': charge,
                'molecule_index': mol_idx
            })
        
        return molecules

**Molecular Properties**:

**Chemical Identity**:
- ``molecule_formula``: Molecular formula (e.g., "C8H10N2O")
- ``molecule_weight``: Molecular weight (g/mol)
- ``molecule_formal_charge``: Net formal charge
- ``molecule_multiplicity``: Spin multiplicity

**Connectivity**:
- ``n_atoms``: Number of atoms in molecule
- ``n_bonds``: Number of covalent bonds
- ``n_fragments``: Number of rigid fragments
- ``n_rings``: Number of ring systems

Atom Level
~~~~~~~~~~

Individual atoms are the fundamental units with coordinates, properties, and connectivity:

.. code-block:: python

    # Atom-level data access
    def get_atomic_data(f, structure_index):
        """Extract atomic coordinates and properties."""
        
        n_atoms = f['n_atoms'][structure_index]
        
        # Atomic coordinates (Cartesian, Å)
        coords = f['atom_coords'][structure_index].reshape(n_atoms, 3)
        
        # Atomic properties
        labels = f['atom_label'][structure_index].astype(str)
        elements = f['atom_element'][structure_index].astype(str)
        
        # Bond connectivity
        bond_indices = f['bond_atom_indices'][structure_index]
        bond_orders = f['bond_order'][structure_index]
        
        return {
            'coordinates': coords,
            'labels': labels,
            'elements': elements,
            'bond_indices': bond_indices,
            'bond_orders': bond_orders
        }

**Atomic Properties**:

**Positional Information**:
- ``atom_coords``: Cartesian coordinates (x, y, z) in Å
- ``atom_frac_coords``: Fractional coordinates relative to unit cell
- ``atom_displacement``: Atomic displacement parameters

**Chemical Information**:
- ``atom_element``: Element symbol (e.g., "C", "N", "O")
- ``atom_label``: Unique atom identifier (e.g., "C1", "N2")
- ``atom_type``: Atom type including hybridization
- ``atom_formal_charge``: Formal charge on atom

**Crystallographic Details**:
- ``atom_site_occupancy``: Site occupancy factor (0-1)
- ``atom_thermal_factor``: Isotropic thermal parameter
- ``atom_symmetry_equivalent``: Symmetry operation indices

Fragment Level
~~~~~~~~~~~~~~

CSA automatically identifies rigid molecular fragments - groups of atoms connected by non-rotatable bonds:

.. code-block:: python

    # Fragment-level analysis
    def analyze_fragments(f, structure_index):
        """Analyze rigid molecular fragments."""
        
        n_fragments = f['n_fragments'][structure_index]
        
        fragments = []
        for frag_idx in range(n_fragments):
            # Fragment identification
            formula = f['fragment_formula'][structure_index][frag_idx].decode()
            atom_indices = f['fragment_atom_indices'][structure_index][frag_idx]
            
            # Geometric properties
            com_coords = f['fragment_com_coords'][structure_index][frag_idx * 3:(frag_idx + 1) * 3]
            inertia_eigvals = f['fragment_inertia_eigvals'][structure_index][frag_idx * 3:(frag_idx + 1) * 3]
            
            # Shape descriptors
            asphericity = inertia_eigvals[2] - 0.5 * (inertia_eigvals[0] + inertia_eigvals[1])
            acylindricity = inertia_eigvals[1] - inertia_eigvals[0]
            
            fragments.append({
                'formula': formula,
                'atom_indices': atom_indices,
                'center_of_mass': com_coords,
                'inertia_eigenvalues': inertia_eigvals,
                'asphericity': asphericity,
                'acylindricity': acylindricity
            })
        
        return fragments

**Fragment Properties**:

**Identity and Composition**:
- ``fragment_formula``: Chemical formula of fragment
- ``fragment_atom_indices``: Indices of atoms in fragment
- ``fragment_n_atoms``: Number of atoms in fragment
- ``fragment_molecular_weight``: Fragment molecular weight

**Geometric Properties**:
- ``fragment_com_coords``: Center of mass coordinates (x, y, z)
- ``fragment_inertia_tensor``: 3×3 inertia tensor
- ``fragment_inertia_eigvals``: Principal moments of inertia
- ``fragment_inertia_eigvecs``: Principal axes of inertia

**Shape Descriptors**:
- ``fragment_radius_gyration``: Radius of gyration
- ``fragment_asphericity``: Deviation from spherical shape
- ``fragment_acylindricity``: Deviation from cylindrical shape
- ``fragment_relative_anisotropy``: Overall shape anisotropy

Interaction Level
~~~~~~~~~~~~~~~~~

CSA identifies and characterizes intermolecular interactions:

.. code-block:: python

    # Intermolecular contact analysis
    def analyze_contacts(f, structure_index):
        """Analyze intermolecular contacts and hydrogen bonds."""
        
        # Intermolecular contacts
        n_contacts = f['inter_cc_n_contacts'][structure_index]
        
        contacts = []
        for contact_idx in range(n_contacts):
            # Contact atoms
            central_atom = f['inter_cc_central_atom'][structure_index][contact_idx].decode()
            contact_atom = f['inter_cc_contact_atom'][structure_index][contact_idx].decode()
            
            # Geometric properties
            length = f['inter_cc_length'][structure_index][contact_idx]
            is_hbond = f['inter_cc_is_hbond'][structure_index][contact_idx]
            
            contacts.append({
                'central_atom': central_atom,
                'contact_atom': contact_atom,
                'length': length,
                'is_hydrogen_bond': bool(is_hbond)
            })
        
        return contacts

**Interaction Types**:

**Close Contacts**:
- ``inter_cc_central_atom``: Central atom in contact
- ``inter_cc_contact_atom``: Contacting atom
- ``inter_cc_length``: Contact distance (Å)
- ``inter_cc_angle``: Contact angle (degrees)

**Hydrogen Bonds**:
- ``inter_cc_is_hbond``: Boolean flag for H-bond classification
- ``hbond_donor_atom``: H-bond donor atom
- ``hbond_acceptor_atom``: H-bond acceptor atom
- ``hbond_dha_angle``: Donor-H-acceptor angle (degrees)

**π-π Interactions**:
- ``pi_pi_distance``: Distance between ring centroids (Å)
- ``pi_pi_angle``: Angle between ring planes (degrees)
- ``pi_pi_offset``: Lateral offset between rings (Å)

Data Types and Storage
----------------------

HDF5 Data Organization
~~~~~~~~~~~~~~~~~~~~~~

CSA uses HDF5's advanced features for efficient data storage:

**Fixed-Length Arrays**: For properties that are the same size across all structures:

.. code-block:: python

    # Fixed-length datasets
    f['z_prime']        # Shape: (n_structures,)
    f['cell_volume']    # Shape: (n_structures,)  
    f['cell_density']   # Shape: (n_structures,)

**Variable-Length Arrays**: For properties that vary by structure:

.. code-block:: python

    # Variable-length datasets
    f['atom_coords']           # Shape: varies per structure
    f['fragment_formula']      # Shape: varies per structure
    f['inter_cc_length']       # Shape: varies per structure

**String Handling**: Efficient storage of text data:

.. code-block:: python

    # String datasets with UTF-8 encoding
    refcodes = f['refcode_list'][...].astype(str)
    space_groups = [f['space_group'][i].decode() for i in range(len(f['refcode_list']))]

Data Access Patterns
~~~~~~~~~~~~~~~~~~~~

**Sequential Access**: For processing all structures:

.. code-block:: python

    def process_all_structures(hdf5_file):
        """Process structures sequentially."""
        
        with h5py.File(hdf5_file, 'r') as f:
            n_structures = len(f['refcode_list'])
            
            for i in range(n_structures):
                refcode = f['refcode_list'][i].decode()
                cell_volume = f['cell_volume'][i]
                
                # Process structure...
                yield refcode, cell_volume

**Random Access**: For accessing specific structures:

.. code-block:: python

    def get_structure_by_refcode(hdf5_file, target_refcode):
        """Find and return data for a specific refcode."""
        
        with h5py.File(hdf5_file, 'r') as f:
            refcodes = f['refcode_list'][...].astype(str)
            
            # Find matching index
            indices = np.where(refcodes == target_refcode)[0]
            if len(indices) == 0:
                return None
                
            idx = indices[0]
            
            # Extract structure data
            structure_data = {
                'refcode': target_refcode,
                'space_group': f['space_group'][idx].decode(),
                'cell_volume': f['cell_volume'][idx],
                'n_atoms': f['n_atoms'][idx]
            }
            
            return structure_data

**Batch Access**: For efficient processing of multiple structures:

.. code-block:: python

    def get_batch_properties(hdf5_file, start_idx, batch_size):
        """Get properties for a batch of structures."""
        
        with h5py.File(hdf5_file, 'r') as f:
            end_idx = min(start_idx + batch_size, len(f['refcode_list']))
            
            batch_data = {
                'refcodes': f['refcode_list'][start_idx:end_idx].astype(str),
                'volumes': f['cell_volume'][start_idx:end_idx],
                'densities': f['cell_density'][start_idx:end_idx],
                'n_atoms': f['n_atoms'][start_idx:end_idx]
            }
            
            return batch_data

Memory Management
-----------------

Efficient Data Loading
~~~~~~~~~~~~~~~~~~~~~~

CSA's data model is designed for memory-efficient processing:

**Lazy Loading**: Only load data when needed:

.. code-block:: python

    def lazy_structure_iterator(hdf5_file):
        """Iterate through structures without loading all data."""
        
        with h5py.File(hdf5_file, 'r') as f:
            n_structures = len(f['refcode_list'])
            
            for i in range(n_structures):
                # Load only needed data for this structure
                yield {
                    'index': i,
                    'refcode': f['refcode_list'][i].decode(),
                    'loader': lambda idx=i: load_structure_data(f, idx)
                }

**Chunked Processing**: Process data in manageable chunks:

.. code-block:: python

    def process_in_chunks(hdf5_file, chunk_size=1000):
        """Process structures in memory-efficient chunks."""
        
        with h5py.File(hdf5_file, 'r') as f:
            n_structures = len(f['refcode_list'])
            
            for start in range(0, n_structures, chunk_size):
                end = min(start + chunk_size, n_structures)
                
                # Load chunk data
                chunk_data = {
                    'refcodes': f['refcode_list'][start:end].astype(str),
                    'volumes': f['cell_volume'][start:end],
                    'densities': f['cell_density'][start:end]
                }
                
                # Process chunk
                yield chunk_data

**Selective Loading**: Load only required datasets:

.. code-block:: python

    def load_specific_properties(hdf5_file, properties):
        """Load only specified properties to save memory."""
        
        data = {}
        
        with h5py.File(hdf5_file, 'r') as f:
            for prop in properties:
                if prop in f:
                    data[prop] = f[prop][...]
                else:
                    print(f"Warning: Property '{prop}' not found in dataset")
        
        return data

Compression and Storage Efficiency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CSA uses compression to minimize storage requirements:

**Compression Settings**:

.. code-block:: python

    # HDF5 compression configuration used by CSA
    compression_settings = {
        'compression': 'lz4',      # Fast compression/decompression
        'compression_opts': 4,     # Moderate compression level
        'shuffle': True,           # Byte shuffling for better compression
        'fletcher32': True         # Checksums for data integrity
    }

**Storage Efficiency Statistics**:

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Data Type
     - Uncompressed Size
     - Compressed Size
     - Compression Ratio
   * - Atomic coordinates
     - 12 MB/1000 structures
     - 3.2 MB
     - 3.8:1
   * - Fragment properties
     - 8 MB/1000 structures
     - 2.1 MB
     - 3.8:1
   * - Contact data
     - 15 MB/1000 structures
     - 4.2 MB
     - 3.6:1
   * - String data
     - 2 MB/1000 structures
     - 0.8 MB
     - 2.5:1

Data Validation and Integrity
-----------------------------

Built-in Validation
~~~~~~~~~~~~~~~~~~~

CSA includes comprehensive data validation:

**Data Type Validation**:

.. code-block:: python

    def validate_data_types(hdf5_file):
        """Validate data types in HDF5 file."""
        
        expected_types = {
            'refcode_list': 'string',
            'z_prime': 'int8',
            'cell_volume': 'float32',
            'cell_density': 'float32',
            'n_atoms': 'int16',
            'atom_coords': 'float32'
        }
        
        issues = []
        
        with h5py.File(hdf5_file, 'r') as f:
            for dataset_name, expected_type in expected_types.items():
                if dataset_name in f:
                    actual_type = str(f[dataset_name].dtype)
                    if expected_type not in actual_type:
                        issues.append(f"{dataset_name}: expected {expected_type}, got {actual_type}")
        
        return issues

**Range Validation**:

.. code-block:: python

    def validate_data_ranges(hdf5_file):
        """Validate that data values are within reasonable ranges."""
        
        validation_rules = {
            'cell_volume': (10.0, 50000.0),        # Å³
            'cell_density': (0.5, 5.0),            # g/cm³
            'z_prime': (1, 8),                     # Typical Z' values
            'temperature': (4.0, 1000.0),          # K
            'resolution': (0.5, 5.0)               # Å
        }
        
        issues = []
        
        with h5py.File(hdf5_file, 'r') as f:
            for property_name, (min_val, max_val) in validation_rules.items():
                if property_name in f:
                    data = f[property_name][...]
                    
                    if np.any(data < min_val) or np.any(data > max_val):
                        out_of_range = np.sum((data < min_val) | (data > max_val))
                        issues.append(f"{property_name}: {out_of_range} values out of range [{min_val}, {max_val}]")
        
        return issues

**Consistency Checks**:

.. code-block:: python

    def validate_data_consistency(hdf5_file):
        """Check for internal consistency in the data."""
        
        issues = []
        
        with h5py.File(hdf5_file, 'r') as f:
            n_structures = len(f['refcode_list'])
            
            # Check that all fixed-length arrays have correct size
            fixed_length_datasets = ['z_prime', 'cell_volume', 'cell_density', 'n_atoms']
            for dataset_name in fixed_length_datasets:
                if dataset_name in f:
                    if len(f[dataset_name]) != n_structures:
                        issues.append(f"{dataset_name}: length {len(f[dataset_name])} != {n_structures}")
            
            # Check atom count consistency
            for i in range(min(100, n_structures)):  # Check first 100 structures
                expected_atoms = f['n_atoms'][i]
                actual_coords = len(f['atom_coords'][i]) // 3  # 3 coordinates per atom
                
                if expected_atoms != actual_coords:
                    refcode = f['refcode_list'][i].decode()
                    issues.append(f"{refcode}: n_atoms={expected_atoms} but {actual_coords} coordinates")
        
        return issues

Working with CSA Data
---------------------

Common Analysis Patterns
~~~~~~~~~~~~~~~~~~~~~~~~

**Property Extraction and Analysis**:

.. code-block:: python

    import pandas as pd
    import numpy as np

    def extract_crystal_properties(hdf5_file):
        """Extract crystal properties into a pandas DataFrame."""
        
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

**Fragment Analysis Workflow**:

.. code-block:: python

    def analyze_fragment_shapes(hdf5_file):
        """Analyze molecular fragment shapes across dataset."""
        
        fragment_data = []
        
        with h5py.File(hdf5_file, 'r') as f:
            n_structures = len(f['refcode_list'])
            
            for i in range(n_structures):
                refcode = f['refcode_list'][i].decode()
                n_frags = f['n_fragments'][i]
                
                for j in range(n_frags):
                    # Extract fragment properties
                    formula = f['fragment_formula'][i][j].decode()
                    
                    # Inertia eigenvalues for shape analysis
                    inertia_start = j * 3
                    inertia_end = (j + 1) * 3
                    eigvals = f['fragment_inertia_eigvals'][i][inertia_start:inertia_end]
                    
                    # Calculate shape descriptors
                    asphericity = eigvals[2] - 0.5 * (eigvals[0] + eigvals[1])
                    acylindricity = eigvals[1] - eigvals[0]
                    
                    fragment_data.append({
                        'refcode': refcode,
                        'fragment_formula': formula,
                        'asphericity': asphericity,
                        'acylindricity': acylindricity
                    })
        
        return pd.DataFrame(fragment_data)

**Contact Network Analysis**:

.. code-block:: python

    def build_contact_network(hdf5_file, distance_cutoff=3.5):
        """Build intermolecular contact networks."""
        
        networks = {}
        
        with h5py.File(hdf5_file, 'r') as f:
            n_structures = len(f['refcode_list'])
            
            for i in range(n_structures):
                refcode = f['refcode_list'][i].decode()
                n_contacts = f['inter_cc_n_contacts'][i]
                
                # Build contact list for this structure
                contacts = []
                for j in range(n_contacts):
                    central = f['inter_cc_central_atom'][i][j].decode()
                    contact = f['inter_cc_contact_atom'][i][j].decode()
                    length = f['inter_cc_length'][i][j]
                    is_hbond = f['inter_cc_is_hbond'][i][j]
                    
                    if length <= distance_cutoff:
                        contacts.append({
                            'central': central,
                            'contact': contact,
                            'length': length,
                            'is_hbond': bool(is_hbond)
                        })
                
                networks[refcode] = contacts
        
        return networks

Data Export and Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Export to Common Formats**:

.. code-block:: python

    def export_to_csv(hdf5_file, output_prefix):
        """Export CSA data to CSV files for external analysis."""
        
        # Crystal properties
        crystal_df = extract_crystal_properties(hdf5_file)
        crystal_df.to_csv(f'{output_prefix}_crystals.csv', index=False)
        
        # Fragment properties
        fragment_df = analyze_fragment_shapes(hdf5_file)
        fragment_df.to_csv(f'{output_prefix}_fragments.csv', index=False)
        
        # Contact summary
        contact_summary = []
        with h5py.File(hdf5_file, 'r') as f:
            for i in range(len(f['refcode_list'])):
                refcode = f['refcode_list'][i].decode()
                n_contacts = f['inter_cc_n_contacts'][i]
                n_hbonds = np.sum(f['inter_cc_is_hbond'][i][:n_contacts])
                
                contact_summary.append({
                    'refcode': refcode,
                    'total_contacts': n_contacts,
                    'hydrogen_bonds': n_hbonds,
                    'other_contacts': n_contacts - n_hbonds
                })
        
        pd.DataFrame(contact_summary).to_csv(f'{output_prefix}_contacts.csv', index=False)

**Integration with Analysis Libraries**:

.. code-block:: python

    def prepare_for_sklearn(hdf5_file, properties):
        """Prepare data for scikit-learn analysis."""
        
        X = []
        structure_ids = []
        
        with h5py.File(hdf5_file, 'r') as f:
            n_structures = len(f['refcode_list'])
            
            for i in range(n_structures):
                # Extract specified properties as feature vector
                features = []
                for prop in properties:
                    if prop in f:
                        features.append(f[prop][i])
                
                if len(features) == len(properties):  # Only include complete cases
                    X.append(features)
                    structure_ids.append(f['refcode_list'][i].decode())
        
        return np.array(X), structure_ids

Best Practices
--------------

Data Access Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use batch processing** for large datasets
2. **Close files promptly** to free resources
3. **Cache frequently accessed data** in memory
4. **Use appropriate data types** to minimize memory usage
5. **Leverage HDF5 chunking** for better I/O performance

Error Handling
~~~~~~~~~~~~~~

.. code-block:: python

    def robust_data_access(hdf5_file, dataset_name, structure_index):
        """Safely access data with proper error handling."""
        
        try:
            with h5py.File(hdf5_file, 'r') as f:
                if dataset_name not in f:
                    raise KeyError(f"Dataset '{dataset_name}' not found")
                
                if structure_index >= len(f['refcode_list']):
                    raise IndexError(f"Structure index {structure_index} out of range")
                
                return f[dataset_name][structure_index]
                
        except (OSError, IOError) as e:
            raise RuntimeError(f"Error accessing HDF5 file: {e}")

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Profile data access patterns** before optimization
2. **Use vectorized operations** when possible
3. **Minimize file open/close cycles** in loops
4. **Consider memory vs. speed tradeoffs** for caching
5. **Monitor memory usage** during large analyses

Next Steps
----------

With understanding of CSA's data model:

**Basic Users**: Apply this knowledge in :doc:`basic_analysis` workflows
**Advanced Users**: Explore :doc:`advanced_features` for complex analyses
**Developers**: Review :doc:`../api_reference/io/data_reader` for programmatic access

See Also
--------

:doc:`basic_analysis` : Practical data access examples
:doc:`../examples/reading_data` : Code examples for data access
:doc:`../api_reference/io/hdf5_utils` : Low-level HDF5 utilities
:doc:`../technical_details/performance` : Performance optimization guide