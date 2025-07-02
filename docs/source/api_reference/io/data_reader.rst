data_reader module
==================

.. automodule:: data_reader
   :members:
   :undoc-members:
   :show-inheritance:

HDF5 Raw Data Reading and Batch Loading
---------------------------------------

The ``data_reader`` module provides efficient interfaces for reading raw crystallographic structure data from HDF5 files into memory for batch processing. It handles padding, type conversion, and organization of heterogeneous structural data into uniform arrays suitable for GPU processing.

**Key Features:**

* **Batch data loading** - Read multiple structures simultaneously for efficient processing
* **Automatic padding** - Uniform array dimensions for ragged data (atoms, bonds, contacts)
* **Type conversion** - Convert HDF5 datasets to appropriate NumPy/PyTorch formats
* **Memory optimization** - Efficient data loading patterns for large datasets
* **Error handling** - Robust handling of missing data and format inconsistencies
* **Flexible interfaces** - Support for different data types and structure organizations

RawDataReader Class
-------------------

.. autoclass:: data_reader.RawDataReader

   **Primary Interface for Raw HDF5 Data Access**

   The ``RawDataReader`` class provides methods to read and organize raw structural data from HDF5 files created by the ``StructureDataExtractor``. It handles the complexities of variable-length datasets and converts them into uniform arrays for batch processing.

   **Attributes:**

   * **h5_in** (:obj:`h5py.File`) - Open HDF5 file handle containing structure data under '/structures/<refcode>'

   **Data Organization:**

   The reader expects HDF5 files with the following structure:

   .. code-block:: text

      /structures/
      ├── <refcode1>/
      │   ├── identifier
      │   ├── cell_lengths, cell_angles
      │   ├── atom_label, atom_symbol, atom_coords, ...
      │   ├── bond_atom1_idx, bond_atom2_idx, ...
      │   ├── inter_cc_*, inter_hb_*
      │   └── intra_cc_*, intra_hb_*
      ├── <refcode2>/
      └── ...

   .. automethod:: __init__

      **Initialize Raw Data Reader**

      Parameters:
         * **h5_in** (:obj:`h5py.File`) - Open HDF5 file handle with '/structures' groups

      **Usage Example:**

      .. code-block:: python

         import h5py
         from data_reader import RawDataReader

         # Open HDF5 file
         with h5py.File('structures_raw.h5', 'r') as h5_file:
             reader = RawDataReader(h5_file)
             
             # Use reader methods...
             batch_refcodes = ['AABBCC', 'DDEEGG', 'HHIIJJ']
             crystal_data = reader.read_crystal_parameters(batch_refcodes)

Crystal Parameter Reading
-------------------------

.. automethod:: data_reader.RawDataReader.read_crystal_parameters

   **Load Unit Cell and Crystal Properties**

   Reads crystallographic unit cell parameters and derived properties for a batch of structures.

   **Parameters:**

   * **batch** (:obj:`List[str]`) - Refcode strings for structures to read

   **Returns:**

   * **dict** - Dictionary containing crystal-level data:

     - **cell_lengths** (:obj:`np.ndarray`, shape (B, 3)) - Unit cell lengths [a, b, c] in Ångstroms
     - **cell_angles** (:obj:`np.ndarray`, shape (B, 3)) - Unit cell angles [α, β, γ] in degrees
     - **z_value** (:obj:`np.ndarray`, shape (B,)) - Number of formula units per unit cell
     - **z_prime** (:obj:`np.ndarray`, shape (B,)) - Number of symmetry-independent molecules
     - **cell_volume** (:obj:`np.ndarray`, shape (B,)) - Unit cell volume in Ų
     - **cell_density** (:obj:`np.ndarray`, shape (B,)) - Crystal density in g/cm³
     - **packing_coefficient** (:obj:`np.ndarray`, shape (B,)) - Space-filling efficiency
     - **identifier** (:obj:`np.ndarray`, shape (B,)) - Structure identifiers
     - **space_group** (:obj:`np.ndarray`, shape (B,)) - Space group symbols

   **Usage Example:**

   .. code-block:: python

      # Read crystal parameters for a batch
      batch_refcodes = ['ALANIN', 'GLYCIN', 'BENZEN']
      crystal_data = reader.read_crystal_parameters(batch_refcodes)
      
      print(f"Unit cell volumes: {crystal_data['cell_volume']}")
      print(f"Crystal densities: {crystal_data['cell_density']}")
      print(f"Space groups: {crystal_data['space_group']}")
      
      # Access individual structure data
      for i, refcode in enumerate(batch_refcodes):
          print(f"{refcode}:")
          print(f"  a={crystal_data['cell_lengths'][i, 0]:.2f} Å")
          print(f"  b={crystal_data['cell_lengths'][i, 1]:.2f} Å") 
          print(f"  c={crystal_data['cell_lengths'][i, 2]:.2f} Å")
          print(f"  α={crystal_data['cell_angles'][i, 0]:.1f}°")
          print(f"  β={crystal_data['cell_angles'][i, 1]:.1f}°")
          print(f"  γ={crystal_data['cell_angles'][i, 2]:.1f}°")
          print(f"  Volume: {crystal_data['cell_volume'][i]:.1f} Ų")
          print(f"  Density: {crystal_data['cell_density'][i]:.2f} g/cm³")

Atomic Data Reading
-------------------

.. automethod:: data_reader.RawDataReader.read_atoms

   **Load Padded Atomic Data Arrays**

   Reads atomic coordinates, properties, and connectivity information with automatic padding to uniform dimensions.

   **Parameters:**

   * **batch** (:obj:`List[str]`) - Refcode strings for structures to read
   * **N_max** (:obj:`int`) - Maximum number of atoms for padding

   **Returns:**

   * **dict** - Dictionary containing atom-level data:

     - **atom_label** (:obj:`List[List[str]]`) - Atom labels per structure
     - **atom_symbol** (:obj:`List[List[str]]`) - Element symbols per structure  
     - **atom_coords** (:obj:`np.ndarray`, shape (B, N_max, 3)) - Cartesian coordinates
     - **atom_frac_coords** (:obj:`np.ndarray`, shape (B, N_max, 3)) - Fractional coordinates
     - **atom_weight** (:obj:`np.ndarray`, shape (B, N_max)) - Atomic masses
     - **atom_charge** (:obj:`np.ndarray`, shape (B, N_max)) - Formal charges
     - **atom_sybyl_type** (:obj:`List[List[str]]`) - SYBYL atom types
     - **atom_neighbors** (:obj:`List[List[List[int]]]`) - Connectivity lists
     - **atom_mask** (:obj:`np.ndarray`, shape (B, N_max)) - Valid atom indicators

   **Padding Behavior:**

   * Real atoms are placed at the beginning of each array
   * Padding slots are filled with zeros/empty strings
   * **atom_mask** indicates valid vs. padded positions

   **Usage Example:**

   .. code-block:: python

      # Read atomic data with padding
      max_atoms = 50  # Determined from dimension scanning
      atom_data = reader.read_atoms(batch_refcodes, max_atoms)
      
      print(f"Atomic data shapes:")
      print(f"  Coordinates: {atom_data['atom_coords'].shape}")
      print(f"  Masses: {atom_data['atom_weight'].shape}")
      print(f"  Mask: {atom_data['atom_mask'].shape}")
      
      # Analyze atomic composition
      for i, refcode in enumerate(batch_refcodes):
          mask = atom_data['atom_mask'][i]
          n_atoms = mask.sum()
          symbols = atom_data['atom_symbol'][i][:n_atoms]
          
          print(f"{refcode}: {n_atoms} atoms")
          
          # Count elements
          from collections import Counter
          element_counts = Counter(symbols)
          formula = ''.join(f"{elem}{count}" if count > 1 else elem 
                           for elem, count in sorted(element_counts.items()))
          print(f"  Formula: {formula}")

Bond Data Reading
-----------------

.. automethod:: data_reader.RawDataReader.read_bonds

   **Load Molecular Bond Information**

   Reads bond connectivity, types, and properties with padding for batch processing.

   **Parameters:**

   * **batch** (:obj:`List[str]`) - Refcode strings for structures to read
   * **max_bonds** (:obj:`int`) - Maximum number of bonds for padding

   **Returns:**

   * **dict** - Dictionary containing bond-level data:

     - **bond_atom1_idx** (:obj:`np.ndarray`, shape (B, max_bonds)) - First atom indices
     - **bond_atom2_idx** (:obj:`np.ndarray`, shape (B, max_bonds)) - Second atom indices
     - **bond_type** (:obj:`List[List[str]]`) - Bond type strings ('single', 'double', etc.)
     - **bond_is_rotatable_raw** (:obj:`np.ndarray`, shape (B, max_bonds)) - Raw rotatability flags
     - **bond_is_cyclic** (:obj:`np.ndarray`, shape (B, max_bonds)) - Ring membership flags
     - **bond_length** (:obj:`np.ndarray`, shape (B, max_bonds)) - Bond lengths in Ångstroms
     - **bond_mask** (:obj:`np.ndarray`, shape (B, max_bonds)) - Valid bond indicators

   **Usage Example:**

   .. code-block:: python

      # Read bond data
      max_bonds = 80  # From dimension scanning
      bond_data = reader.read_bonds(batch_refcodes, max_bonds)
      
      # Analyze bond statistics
      for i, refcode in enumerate(batch_refcodes):
          mask = bond_data['bond_mask'][i]
          n_bonds = mask.sum()
          
          if n_bonds > 0:
              bond_types = bond_data['bond_type'][i][:n_bonds]
              bond_lengths = bond_data['bond_length'][i][mask]
              rotatable_bonds = bond_data['bond_is_rotatable_raw'][i][mask].sum()
              
              print(f"{refcode}: {n_bonds} bonds")
              print(f"  Types: {set(bond_types)}")
              print(f"  Length range: {bond_lengths.min():.2f}-{bond_lengths.max():.2f} Å")
              print(f"  Rotatable bonds: {rotatable_bonds}")

Contact Data Reading
--------------------

.. automethod:: data_reader.RawDataReader.read_intermolecular_contacts

   **Load Intermolecular Contact Data**

   Reads intermolecular atomic contacts with symmetry operations and geometric properties.

   **Parameters:**

   * **batch** (:obj:`List[str]`) - Refcode strings for structures to read  
   * **max_contacts** (:obj:`int`) - Maximum number of contacts for padding

   **Returns:**

   * **dict** - Dictionary containing intermolecular contact data:

     - **inter_cc_id** (:obj:`List[List[str]]`) - Contact identifiers
     - **inter_cc_central_atom** (:obj:`List[List[str]]`) - Central atom labels
     - **inter_cc_contact_atom** (:obj:`List[List[str]]`) - Contact atom labels
     - **inter_cc_central_atom_idx** (:obj:`np.ndarray`) - Central atom indices
     - **inter_cc_contact_atom_idx** (:obj:`np.ndarray`) - Contact atom indices
     - **inter_cc_*_coords** (:obj:`np.ndarray`) - Cartesian coordinates
     - **inter_cc_*_frac_coords** (:obj:`np.ndarray`) - Fractional coordinates
     - **inter_cc_length** (:obj:`np.ndarray`) - Contact distances
     - **inter_cc_strength** (:obj:`np.ndarray`) - Contact strength metrics
     - **inter_cc_symmetry** (:obj:`List[List[str]]`) - Symmetry operator strings
     - **inter_cc_in_los** (:obj:`np.ndarray`) - Line-of-sight flags
     - **inter_cc_mask** (:obj:`np.ndarray`) - Valid contact indicators

.. automethod:: data_reader.RawDataReader.read_intermolecular_hbonds

   **Load Intermolecular Hydrogen Bond Data**

   Reads hydrogen bond interactions with donor-hydrogen-acceptor triplet information.

   **Parameters:**

   * **batch** (:obj:`List[str]`) - Refcode strings for structures to read
   * **max_hbonds** (:obj:`int`) - Maximum number of hydrogen bonds for padding

   **Returns:**

   * **dict** - Dictionary containing hydrogen bond data:

     - **inter_hb_id** (:obj:`List[List[str]]`) - H-bond identifiers
     - **inter_hb_central_atom** (:obj:`List[List[str]]`) - Donor atom labels
     - **inter_hb_hydrogen_atom** (:obj:`List[List[str]]`) - Hydrogen atom labels
     - **inter_hb_contact_atom** (:obj:`List[List[str]]`) - Acceptor atom labels
     - **inter_hb_*_idx** (:obj:`np.ndarray`) - Atom indices for donor/H/acceptor
     - **inter_hb_*_coords** (:obj:`np.ndarray`) - Coordinates for all three atoms
     - **inter_hb_length** (:obj:`np.ndarray`) - Donor-acceptor distances
     - **inter_hb_angle** (:obj:`np.ndarray`) - Donor-H-acceptor angles
     - **inter_hb_symmetry** (:obj:`List[List[str]]`) - Symmetry operations
     - **inter_hb_mask** (:obj:`np.ndarray`) - Valid H-bond indicators

   **Usage Example:**

   .. code-block:: python

      # Read intermolecular interactions
      max_contacts = 200
      max_hbonds = 50
      
      contact_data = reader.read_intermolecular_contacts(batch_refcodes, max_contacts)
      hbond_data = reader.read_intermolecular_hbonds(batch_refcodes, max_hbonds)
      
      # Analyze interaction patterns
      for i, refcode in enumerate(batch_refcodes):
          n_contacts = contact_data['inter_cc_mask'][i].sum()
          n_hbonds = hbond_data['inter_hb_mask'][i].sum()
          
          print(f"{refcode}:")
          print(f"  Intermolecular contacts: {n_contacts}")
          print(f"  Hydrogen bonds: {n_hbonds}")
          
          if n_hbonds > 0:
              hb_lengths = hbond_data['inter_hb_length'][i][:n_hbonds]
              hb_angles = hbond_data['inter_hb_angle'][i][:n_hbonds]
              
              print(f"  H-bond lengths: {hb_lengths.mean():.2f} ± {hb_lengths.std():.2f} Å")
              print(f"  H-bond angles: {hb_angles.mean():.1f} ± {hb_angles.std():.1f}°")

Intramolecular Data Reading
---------------------------

.. automethod:: data_reader.RawDataReader.read_intramolecular_contacts

   **Load Intramolecular Contact Data**

   Reads contacts within individual molecules, typically for conformational analysis.

.. automethod:: data_reader.RawDataReader.read_intramolecular_hbonds

   **Load Intramolecular Hydrogen Bond Data**

   Reads hydrogen bonds within individual molecules for internal structure analysis.

   **Usage Pattern:**

   .. code-block:: python

      # Read intramolecular interactions
      intra_contacts = reader.read_intramolecular_contacts(batch_refcodes, max_intra_contacts)
      intra_hbonds = reader.read_intramolecular_hbonds(batch_refcodes, max_intra_hbonds)
      
      # These follow the same data structure as intermolecular versions
      # but analyze internal molecular geometry

Advanced Usage Patterns
------------------------

**Efficient Batch Processing**

.. code-block:: python

   def process_large_dataset_efficiently(h5_file_path, batch_size=32):
       """Process large datasets with memory-efficient batching."""
       
       with h5py.File(h5_file_path, 'r') as h5_file:
           reader = RawDataReader(h5_file)
           
           # Get all refcodes
           all_refcodes = [key for key in h5_file['structures'].keys()]
           n_structures = len(all_refcodes)
           
           print(f"Processing {n_structures} structures in batches of {batch_size}")
           
           # Process in batches
           results = []
           for start in range(0, n_structures, batch_size):
               end = min(start + batch_size, n_structures)
               batch_refcodes = all_refcodes[start:end]
               
               print(f"Processing batch {start//batch_size + 1}: structures {start+1}-{end}")
               
               # Read data for this batch
               crystal_data = reader.read_crystal_parameters(batch_refcodes)
               atom_data = reader.read_atoms(batch_refcodes, max_atoms=100)
               
               # Process data (placeholder for actual analysis)
               batch_results = analyze_batch(crystal_data, atom_data)
               results.extend(batch_results)
           
           return results

**Data Quality Assessment**

.. code-block:: python

   def assess_data_quality(reader, batch_refcodes, max_atoms, max_bonds):
       """Assess quality and completeness of structural data."""
       
       crystal_data = reader.read_crystal_parameters(batch_refcodes)
       atom_data = reader.read_atoms(batch_refcodes, max_atoms)
       bond_data = reader.read_bonds(batch_refcodes, max_bonds)
       
       quality_report = {}
       
       for i, refcode in enumerate(batch_refcodes):
           # Check basic completeness
           n_atoms = atom_data['atom_mask'][i].sum()
           n_bonds = bond_data['bond_mask'][i].sum()
           
           # Check for reasonable values
           cell_volume = crystal_data['cell_volume'][i]
           density = crystal_data['cell_density'][i]
           
           # Assess quality flags
           quality_flags = []
           
           if n_atoms < 5:
               quality_flags.append("too_few_atoms")
           if n_atoms > max_atoms * 0.9:
               quality_flags.append("near_padding_limit") 
           if cell_volume < 100 or cell_volume > 10000:
               quality_flags.append("unusual_volume")
           if density < 0.5 or density > 5.0:
               quality_flags.append("unusual_density")
           if n_bonds < n_atoms - 1:
               quality_flags.append("disconnected_structure")
           
           quality_report[refcode] = {
               'n_atoms': n_atoms,
               'n_bonds': n_bonds,
               'cell_volume': cell_volume,
               'density': density,
               'quality_flags': quality_flags,
               'quality_score': len(quality_flags)  # Lower is better
           }
       
       return quality_report

**Integration with Processing Pipeline**

.. code-block:: python

   def integrated_data_loading(h5_file_path, batch_refcodes, dimensions):
       """Complete data loading for processing pipeline."""
       
       with h5py.File(h5_file_path, 'r') as h5_file:
           reader = RawDataReader(h5_file)
           
           # Load all required data types
           crystal_data = reader.read_crystal_parameters(batch_refcodes)
           atom_data = reader.read_atoms(batch_refcodes, dimensions['atoms'])
           bond_data = reader.read_bonds(batch_refcodes, dimensions['bonds'])
           
           contact_data = reader.read_intermolecular_contacts(
               batch_refcodes, dimensions['contacts_inter']
           )
           hbond_data = reader.read_intermolecular_hbonds(
               batch_refcodes, dimensions['hbonds_inter']
           )
           
           # Convert to tensors for GPU processing
           import torch
           device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
           
           # Convert numeric arrays to tensors
           for key, value in crystal_data.items():
               if isinstance(value, np.ndarray) and value.dtype.kind in 'if':
                   crystal_data[key] = torch.from_numpy(value).to(device)
           
           for key, value in atom_data.items():
               if isinstance(value, np.ndarray) and value.dtype.kind in 'ifb':
                   atom_data[key] = torch.from_numpy(value).to(device)
           
           # Similar conversion for other data types...
           
           return {
               'crystal': crystal_data,
               'atoms': atom_data,
               'bonds': bond_data,
               'contacts': contact_data,
               'hbonds': hbond_data
           }

Error Handling and Diagnostics
------------------------------

**Missing Data Handling**

.. code-block:: python

   def robust_data_reading(reader, batch_refcodes, dimensions):
       """Robust data reading with error handling."""
       
       successful_reads = []
       failed_reads = []
       
       for refcode in batch_refcodes:
           try:
               # Try to read each structure individually first
               crystal_data = reader.read_crystal_parameters([refcode])
               atom_data = reader.read_atoms([refcode], dimensions['atoms'])
               
               # Check for required fields
               required_fields = ['cell_lengths', 'cell_angles', 'atom_coords']
               for field in required_fields:
                   if field not in crystal_data and field not in atom_data:
                       raise KeyError(f"Missing required field: {field}")
               
               successful_reads.append(refcode)
               
           except Exception as e:
               print(f"Warning: Failed to read {refcode}: {e}")
               failed_reads.append((refcode, str(e)))
       
       if failed_reads:
           print(f"Failed to read {len(failed_reads)} structures:")
           for refcode, error in failed_reads:
               print(f"  {refcode}: {error}")
       
       # Process only successful reads
       if successful_reads:
           return load_successful_structures(reader, successful_reads, dimensions)
       else:
           raise ValueError("No structures could be read successfully")

**Data Validation**

.. code-block:: python

   def validate_loaded_data(data_dict):
       """Validate loaded data for consistency and completeness."""
       
       print("Data Validation Report:")
       
       # Check crystal data
       crystal_data = data_dict['crystal']
       n_structures = len(crystal_data['cell_lengths'])
       
       print(f"  Loaded {n_structures} structures")
       
       # Validate cell parameters
       lengths = crystal_data['cell_lengths']
       angles = crystal_data['cell_angles']
       
       if np.any(lengths <= 0):
           print("  Warning: Non-positive cell lengths detected")
       
       if np.any((angles <= 0) | (angles >= 180)):
           print("  Warning: Invalid cell angles detected")
       
       # Check atomic data consistency
       atom_data = data_dict['atoms']
       atom_coords = atom_data['atom_coords']
       atom_mask = atom_data['atom_mask']
       
       # Verify mask consistency
       for i in range(n_structures):
           n_atoms = atom_mask[i].sum()
           valid_coords = atom_coords[i][atom_mask[i]]
           
           if np.any(np.isnan(valid_coords)):
               print(f"  Warning: NaN coordinates in structure {i}")
           
           if np.any(np.abs(valid_coords) > 1000):
               print(f"  Warning: Unusually large coordinates in structure {i}")
       
       print("  Validation complete")

Performance Optimization
------------------------

**Memory-Efficient Loading**

.. code-block:: python

   def memory_efficient_data_loading(h5_file_path, total_structures, batch_size=32):
       """Load data with careful memory management."""
       
       import psutil
       import gc
       
       def get_memory_usage():
           return psutil.Process().memory_info().rss / 1024 / 1024  # MB
       
       initial_memory = get_memory_usage()
       print(f"Initial memory usage: {initial_memory:.1f} MB")
       
       with h5py.File(h5_file_path, 'r') as h5_file:
           reader = RawDataReader(h5_file)
           all_refcodes = list(h5_file['structures'].keys())[:total_structures]
           
           results = []
           
           for start in range(0, len(all_refcodes), batch_size):
               batch_refcodes = all_refcodes[start:start+batch_size]
               
               # Load batch data
               crystal_data = reader.read_crystal_parameters(batch_refcodes)
               
               # Process immediately to avoid memory accumulation
               batch_results = process_crystal_data(crystal_data)
               results.extend(batch_results)
               
               # Clear references and force garbage collection
               del crystal_data
               gc.collect()
               
               current_memory = get_memory_usage()
               print(f"Batch {start//batch_size + 1}: {current_memory:.1f} MB "
                     f"(+{current_memory - initial_memory:.1f} MB)")
       
       return results

Cross-References
----------------

**Related CSA Modules:**

* :doc:`../io/data_writer` - Writing processed data back to HDF5
* :doc:`../io/dimension_scanner` - Determining optimal padding dimensions
* :doc:`../extraction/structure_data_extractor` - Creating raw HDF5 files
* :doc:`../extraction/structure_post_extraction_processor` - Using loaded data for processing

**External Dependencies:**

* `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_ - Hierarchical data format
* `h5py <https://www.h5py.org/>`_ - Python interface to HDF5
* `NumPy <https://numpy.org/>`_ - Array operations and data structures

**File Format References:**

* HDF5 User Guide: https://docs.hdfgroup.org/hdf5/develop/_u_g.html
* h5py Documentation: https://docs.h5py.org/en/stable/