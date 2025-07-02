data_writer module
==================

.. automodule:: data_writer
   :members:
   :undoc-members:
   :show-inheritance:

HDF5 Data Writing and Storage Management
----------------------------------------

The ``data_writer`` module provides efficient interfaces for writing both raw and computed structural data to HDF5 files. It handles the complexities of variable-length datasets, batch writing operations, and proper data type management for crystallographic structure analysis pipelines.

**Key Features:**

* **Dual writer classes** - Separate interfaces for raw and computed data
* **Batch writing operations** - Efficient slice-by-slice data storage
* **Variable-length datasets** - Proper handling of ragged arrays (atoms, bonds, contacts)
* **Type management** - Automatic conversion between PyTorch tensors and NumPy arrays
* **Memory optimization** - Efficient writing patterns for large datasets
* **Data integrity** - Consistent formatting and validation during writes

Writer Classes Overview
-----------------------

The module provides two main writer classes:

* **RawDataWriter** - Handles raw structural data from extraction pipeline
* **ComputedDataWriter** - Handles processed features and computed descriptors

Both classes follow similar patterns but handle different data types and formatting requirements.

RawDataWriter Class
-------------------

.. autoclass:: data_writer.RawDataWriter

   **Raw Structural Data Writing Interface**

   The ``RawDataWriter`` class handles writing raw crystallographic data extracted from CSD structures into HDF5 files with proper formatting and organization for downstream processing.

   **Attributes:**

   * **h5_out** (:obj:`h5py.File`) - Open HDF5 file handle for writing processed data

   **Data Organization Pattern:**

   Raw data is organized into distinct categories:
   - Crystal-level parameters (fixed-size arrays)
   - Atomic data (variable-length with padding)
   - Bond connectivity (variable-length with padding)
   - Intermolecular interactions (contacts and H-bonds)
   - Intramolecular interactions (contacts and H-bonds)

   .. automethod:: __init__

      **Initialize Raw Data Writer**

      Parameters:
         * **h5_out** (:obj:`h5py.File`) - Open HDF5 file for writing processed data

      **Usage Example:**

      .. code-block:: python

         import h5py
         from data_writer import RawDataWriter

         # Create or open output file
         with h5py.File('structures_processed.h5', 'w') as h5_out:
             writer = RawDataWriter(h5_out)
             
             # Write data in batches...
             writer.write_raw_crystal_data(start_index, crystal_parameters)

Crystal Data Writing
--------------------

.. automethod:: data_writer.RawDataWriter.write_raw_crystal_data

   **Write Crystal-Level Parameters**

   Stores unit cell parameters and crystal properties in fixed-size datasets.

   **Parameters:**

   * **start** (:obj:`int`) - Index offset in output datasets for this batch
   * **crystal_parameters** (:obj:`Dict[str, Any]`) - Dictionary of crystal-level data including:

     - **cell_lengths** - Unit cell parameters [a, b, c]
     - **cell_angles** - Unit cell angles [α, β, γ]  
     - **z_value** - Number of formula units per unit cell
     - **z_prime** - Number of symmetry-independent molecules
     - **cell_volume** - Unit cell volume
     - **cell_density** - Crystal density
     - **space_group** - Space group symbol
     - **identifier** - Structure identifier

   **Data Types and Conversion:**

   The method automatically handles conversion between different data formats:

   .. code-block:: python

      # Automatic tensor to array conversion
      if torch.is_tensor(value):
          array_data = value.detach().cpu().numpy()
      else:
          array_data = np.asarray(value)
      
      # Write to appropriate dataset slice
      h5_dataset[start:start+batch_size] = array_data

   **Usage Example:**

   .. code-block:: python

      # Prepare crystal parameter data
      crystal_params = {
          'cell_lengths': np.array([[5.0, 5.0, 5.0], [6.0, 8.0, 10.0]]),
          'cell_angles': np.array([[90.0, 90.0, 90.0], [75.0, 85.0, 95.0]]),
          'cell_volume': np.array([125.0, 450.2]),
          'cell_density': np.array([1.45, 1.32]),
          'space_group': ['P1', 'P-1']
      }
      
      # Write to HDF5 starting at structure index 10
      writer.write_raw_crystal_data(start=10, crystal_parameters=crystal_params)

Atomic Data Writing
-------------------

.. automethod:: data_writer.RawDataWriter.write_raw_atom_data

   **Write Padded Atomic Information**

   Stores atomic coordinates, properties, and connectivity in variable-length datasets with proper padding management.

   **Parameters:**

   * **start** (:obj:`int`) - Index offset in output datasets for this batch
   * **atom_parameters** (:obj:`Dict[str, Any]`) - Dictionary containing atomic data:

     - **atom_label** - Atom labels per structure
     - **atom_symbol** - Element symbols
     - **atom_coords** - Cartesian coordinates (Å)
     - **atom_frac_coords** - Fractional coordinates
     - **atom_weight** - Atomic masses (Da)
     - **atom_charge** - Formal charges
     - **atom_sybyl_type** - SYBYL atom types
     - **atom_neighbors** - Connectivity information
     - **atom_mask** - Valid atom indicators

   **Variable-Length Data Handling:**

   The method handles variable-length datasets using HDF5's VLen (Variable Length) data types:

   .. code-block:: python

      # For each structure in the batch
      for i, refcode in enumerate(batch):
          # Get number of real atoms
          n_atoms = atom_mask[i].sum()
          
          # Extract valid data only
          valid_coords = atom_coords[i, :n_atoms, :].flatten()  # (n_atoms * 3,)
          valid_symbols = atom_symbols[i][:n_atoms]
          
          # Write to variable-length datasets
          h5_dataset['atom_coords'][structure_index] = valid_coords
          h5_dataset['atom_symbol'][structure_index] = valid_symbols

   **Usage Example:**

   .. code-block:: python

      # Prepare atomic data with proper padding
      batch_size = 3
      max_atoms = 50
      
      atom_params = {
          'atom_coords': np.random.rand(batch_size, max_atoms, 3),
          'atom_symbol': [['C', 'C', 'H', 'H'], ['N', 'O', 'H'], ['C', 'O']],
          'atom_weight': np.random.rand(batch_size, max_atoms),
          'atom_mask': np.array([[True, True, True, True] + [False]*46,
                                [True, True, True] + [False]*47,
                                [True, True] + [False]*48])
      }
      
      writer.write_raw_atom_data(start=0, atom_parameters=atom_params)

Bond Data Writing
-----------------

.. automethod:: data_writer.RawDataWriter.write_raw_bond_data

   **Write Molecular Bond Information**

   Stores bond connectivity, types, and properties with variable-length formatting.

   **Parameters:**

   * **start** (:obj:`int`) - Index offset for this batch
   * **bond_parameters** (:obj:`Dict[str, Any]`) - Dictionary containing bond data:

     - **bond_atom1_idx** - First atom indices
     - **bond_atom2_idx** - Second atom indices
     - **bond_type** - Bond type strings
     - **bond_is_rotatable_raw** - Rotatability flags
     - **bond_is_cyclic** - Ring membership
     - **bond_length** - Bond lengths
     - **bond_mask** - Valid bond indicators

   **Usage Example:**

   .. code-block:: python

      bond_params = {
          'bond_atom1_idx': np.array([[0, 1, 2], [0, 1, 2, 3]]),
          'bond_atom2_idx': np.array([[1, 2, 3], [1, 2, 3, 4]]),
          'bond_type': [['single', 'double', 'single'], ['single', 'single', 'aromatic', 'single']],
          'bond_length': np.array([[1.54, 1.34, 1.45], [1.47, 1.42, 1.39, 1.51]]),
          'bond_mask': np.array([[True, True, True, False], [True, True, True, True]])
      }
      
      writer.write_raw_bond_data(start=0, bond_parameters=bond_params)

Contact Data Writing
--------------------

.. automethod:: data_writer.RawDataWriter.write_raw_intermolecular_contact_data

   **Write Intermolecular Contact Information**

   Stores intermolecular atomic contacts with symmetry operations and geometric properties.

.. automethod:: data_writer.RawDataWriter.write_raw_intermolecular_hbond_data

   **Write Intermolecular Hydrogen Bond Data**

   Stores hydrogen bond interactions with donor-hydrogen-acceptor information.

.. automethod:: data_writer.RawDataWriter.write_raw_intramolecular_contact_data

   **Write Intramolecular Contact Information**

   Stores contacts within individual molecules.

.. automethod:: data_writer.RawDataWriter.write_raw_intramolecular_hbond_data

   **Write Intramolecular Hydrogen Bond Data**

   Stores internal molecular hydrogen bonds.

ComputedDataWriter Class
------------------------

.. autoclass:: data_writer.ComputedDataWriter

   **Computed Feature Data Writing Interface**

   The ``ComputedDataWriter`` class handles writing processed features and computed descriptors derived from raw structural data.

   **Key Differences from RawDataWriter:**

   * Handles computed tensors and derived features
   * Different data organization patterns
   * Optimized for machine learning-ready formats
   * Includes fragment-level and interaction-level features

   .. automethod:: __init__

      **Initialize Computed Data Writer**

      Parameters:
         * **h5_out** (:obj:`h5py.File`) - Open HDF5 file for writing processed data

Computed Data Writing Methods
-----------------------------

.. automethod:: data_writer.ComputedDataWriter.write_computed_crystal_data

   **Write Computed Crystal Features**

   Stores derived crystal properties and transformation matrices.

   **Parameters:**

   * **start** (:obj:`int`) - Index offset for this batch
   * **crystal_parameters** (:obj:`Dict[str, Any]`) - Dictionary containing:

     - **scaled_cell** (:obj:`torch.Tensor`, shape (B, 6)) - Normalized cell parameters
     - **cell_matrix** (:obj:`torch.Tensor`, shape (B, 3, 3)) - Transformation matrices

   **Tensor Conversion:**

   .. code-block:: python

      # Automatic PyTorch tensor conversion
      for key, vals in crystal_parameters.items():
          if torch.is_tensor(vals):
              arrays[key] = vals.detach().cpu().numpy()
          else:
              arrays[key] = np.asarray(vals)
      
      # Write different dimensionalities appropriately
      if array.ndim == 2:
          dataset[start:start+B, :] = array
      elif array.ndim == 3:
          dataset[start:start+B, :, :] = array

   **Usage Example:**

   .. code-block:: python

      import torch
      
      # Computed crystal features
      computed_crystal = {
          'scaled_cell': torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                      [1.0, 1.33, 1.67, 0.83, 0.94, 1.06]]),
          'cell_matrix': torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
      }
      
      writer.write_computed_crystal_data(start=0, crystal_parameters=computed_crystal)

.. automethod:: data_writer.ComputedDataWriter.write_computed_atom_data

   **Write Computed Atomic Features**

   Stores atom-level computed properties and descriptors.

   **Parameters:**

   * **start** (:obj:`int`) - Index offset for this batch
   * **atom_parameters** (:obj:`Dict[str, Any]`) - Dictionary containing:

     - **atom_fragment_id** - Fragment assignments
     - **atom_dist_to_special_planes** - Distances to crystallographic planes

.. automethod:: data_writer.ComputedDataWriter.write_computed_bond_data

   **Write Computed Bond Features**

   Stores bond-level computed properties and geometric descriptors.

.. automethod:: data_writer.ComputedDataWriter.write_computed_molecule_data

   **Write Computed Molecular Features**

   Stores intramolecular geometric features like bond angles and torsions.

.. automethod:: data_writer.ComputedDataWriter.write_computed_fragment_data

   **Write Fragment-Level Features**

   Stores molecular fragment properties and descriptors.

   **Parameters:**

   * **start** (:obj:`int`) - Index offset for this batch
   * **fragment_parameters** (:obj:`Dict[str, Any]`) - Dictionary containing extensive fragment data:

     - **n_fragments** - Number of fragments per structure
     - **fragment_local_id** - Local fragment identifiers
     - **fragment_formula** - Molecular formulas
     - **fragment_n_atoms** - Atom counts per fragment
     - **fragment_com_coords** - Centers of mass
     - **fragment_inertia_tensors** - Moment of inertia tensors
     - **fragment_inertia_eigvals** - Principal moments
     - **fragment_inertia_eigvecs** - Principal axes
     - **fragment_quaternions** - Orientation quaternions
     - **fragment_Ql** - Steinhardt order parameters
     - **fragment_planarity_metrics** - Shape descriptors

   **Fragment Data Organization:**

   .. code-block:: python

      # Example fragment parameter structure
      fragment_params = {
          'n_fragments': [2, 3, 1],  # Number of fragments per structure
          'fragment_formula': [['C6H6', 'H2O'], ['C2H4', 'NH3', 'CO2'], ['C12H10']],
          'fragment_com_coords': np.array([...]),  # Centers of mass
          'fragment_inertia_eigvals': np.array([...]),  # Principal moments
          'fragment_planarity_rmsd': np.array([...])  # Planarity metrics
      }

Advanced Writing Patterns
--------------------------

**Batch Processing Workflow**

.. code-block:: python

   def complete_data_writing_workflow(h5_out_path, processed_data_batches):
       """Complete workflow for writing processed structural data."""
       
       with h5py.File(h5_out_path, 'w') as h5_out:
           raw_writer = RawDataWriter(h5_out)
           computed_writer = ComputedDataWriter(h5_out)
           
           total_written = 0
           
           for batch_idx, batch_data in enumerate(processed_data_batches):
               batch_size = len(batch_data['refcodes'])
               start_idx = total_written
               
               print(f"Writing batch {batch_idx + 1}: structures {start_idx + 1}-{start_idx + batch_size}")
               
               # Write raw data
               raw_writer.write_raw_crystal_data(start_idx, batch_data['raw_crystal'])
               raw_writer.write_raw_atom_data(start_idx, batch_data['raw_atoms'])
               raw_writer.write_raw_bond_data(start_idx, batch_data['raw_bonds'])
               
               # Write computed features
               computed_writer.write_computed_crystal_data(start_idx, batch_data['computed_crystal'])
               computed_writer.write_computed_atom_data(start_idx, batch_data['computed_atoms'])
               computed_writer.write_computed_fragment_data(start_idx, batch_data['computed_fragments'])
               
               total_written += batch_size
           
           print(f"Successfully wrote {total_written} structures to {h5_out_path}")

**Memory-Efficient Writing**

.. code-block:: python

   def memory_efficient_writing(data_generator, h5_out_path, batch_size=32):
       """Write data efficiently without accumulating large arrays in memory."""
       
       with h5py.File(h5_out_path, 'w') as h5_out:
           raw_writer = RawDataWriter(h5_out)
           computed_writer = ComputedDataWriter(h5_out)
           
           current_batch = []
           start_idx = 0
           
           for structure_data in data_generator:
               current_batch.append(structure_data)
               
               # Write when batch is full
               if len(current_batch) >= batch_size:
                   batch_data = organize_batch_data(current_batch)
                   
                   # Write and immediately clear batch
                   write_batch(raw_writer, computed_writer, start_idx, batch_data)
                   
                   start_idx += len(current_batch)
                   current_batch.clear()
                   
                   # Force garbage collection
                   import gc
                   gc.collect()
           
           # Write remaining partial batch
           if current_batch:
               batch_data = organize_batch_data(current_batch)
               write_batch(raw_writer, computed_writer, start_idx, batch_data)

**Error Handling and Validation**

.. code-block:: python

   def robust_data_writing(h5_out_path, data_batches):
       """Write data with comprehensive error handling."""
       
       with h5py.File(h5_out_path, 'w') as h5_out:
           raw_writer = RawDataWriter(h5_out)
           computed_writer = ComputedDataWriter(h5_out)
           
           successful_writes = 0
           failed_writes = []
           
           for batch_idx, batch_data in enumerate(data_batches):
               try:
                   # Validate data before writing
                   validate_batch_data(batch_data)
                   
                   # Attempt to write
                   start_idx = successful_writes
                   raw_writer.write_raw_crystal_data(start_idx, batch_data['raw_crystal'])
                   computed_writer.write_computed_crystal_data(start_idx, batch_data['computed_crystal'])
                   
                   successful_writes += len(batch_data['refcodes'])
                   print(f"✓ Batch {batch_idx + 1}: {len(batch_data['refcodes'])} structures written")
                   
               except Exception as e:
                   error_msg = f"Failed to write batch {batch_idx + 1}: {str(e)}"
                   print(f"✗ {error_msg}")
                   failed_writes.append((batch_idx, error_msg))
                   
                   # Continue with next batch
                   continue
           
           print(f"\nWriting summary:")
           print(f"  Successful: {successful_writes} structures")
           print(f"  Failed: {len(failed_writes)} batches")
           
           if failed_writes:
               print("Failed batches:")
               for batch_idx, error in failed_writes:
                   print(f"  Batch {batch_idx + 1}: {error}")

   def validate_batch_data(batch_data):
       """Validate batch data before writing."""
       
       # Check required keys
       required_keys = ['refcodes', 'raw_crystal', 'computed_crystal']
       for key in required_keys:
           if key not in batch_data:
               raise ValueError(f"Missing required key: {key}")
       
       # Check data consistency
       n_structures = len(batch_data['refcodes'])
       
       # Validate crystal data shapes
       crystal_data = batch_data['raw_crystal']
       if 'cell_lengths' in crystal_data:
           if crystal_data['cell_lengths'].shape[0] != n_structures:
               raise ValueError("Crystal data length mismatch")
       
       # Check for NaN values
       for key, value in crystal_data.items():
           if isinstance(value, np.ndarray) and np.any(np.isnan(value)):
               raise ValueError(f"NaN values detected in {key}")

Data Type Management
--------------------

**Automatic Type Conversion**

.. code-block:: python

   def convert_data_types(data_dict):
       """Convert data types for optimal HDF5 storage."""
       
       converted = {}
       
       for key, value in data_dict.items():
           if torch.is_tensor(value):
               # PyTorch tensor to NumPy array
               converted[key] = value.detach().cpu().numpy()
           
           elif isinstance(value, list):
               # Handle nested lists (e.g., atom labels)
               if all(isinstance(item, list) for item in value):
                   converted[key] = value  # Keep as nested list
               else:
                   converted[key] = np.array(value)
           
           elif isinstance(value, np.ndarray):
               # Ensure appropriate dtype
               if value.dtype == np.float64:
                   converted[key] = value.astype(np.float32)  # Reduce precision
               elif value.dtype == np.int64:
                   converted[key] = value.astype(np.int32)    # Reduce size
               else:
                   converted[key] = value
           
           else:
               converted[key] = value
       
       return converted

**HDF5 Dataset Configuration**

.. code-block:: python

   def configure_hdf5_datasets(h5_file, n_structures, dimensions):
       """Configure HDF5 datasets with optimal settings."""
       
       # Compression and chunking settings
       compression_opts = {
           'compression': 'gzip',
           'compression_opts': 6,
           'shuffle': True,
           'fletcher32': True
       }
       
       # Create datasets with appropriate shapes and types
       crystal_datasets = {
           'cell_lengths': (n_structures, 3, np.float32),
           'cell_angles': (n_structures, 3, np.float32),
           'cell_volume': (n_structures, np.float32),
           'scaled_cell': (n_structures, 6, np.float32),
           'cell_matrix': (n_structures, 3, 3, np.float32)
       }
       
       for name, (shape, dtype) in crystal_datasets.items():
           h5_file.create_dataset(
               name, 
               shape=shape if isinstance(shape, tuple) else (shape,),
               dtype=dtype,
               chunks=True,
               **compression_opts
           )
       
       # Variable-length datasets for ragged arrays
       vlen_datasets = {
           'atom_coords': h5py.vlen_dtype(np.float32),
           'atom_symbol': h5py.string_dtype('utf-8'),
           'bond_type': h5py.string_dtype('utf-8')
       }
       
       for name, dtype in vlen_datasets.items():
           h5_file.create_dataset(
               name,
               shape=(n_structures,),
               dtype=dtype,
               **compression_opts
           )

Performance Optimization
------------------------

**Chunking and Compression**

.. code-block:: python

   def optimize_hdf5_performance(h5_file, dataset_name, data_shape):
       """Optimize HDF5 dataset for read/write performance."""
       
       # Calculate optimal chunk size
       if len(data_shape) == 1:
           # 1D datasets: chunk by batch size
           chunk_shape = (min(1000, data_shape[0]),)
       elif len(data_shape) == 2:
           # 2D datasets: chunk by row
           chunk_shape = (1, data_shape[1])
       elif len(data_shape) == 3:
           # 3D datasets: chunk by individual matrices
           chunk_shape = (1, data_shape[1], data_shape[2])
       else:
           chunk_shape = None
       
       # Configure dataset with optimal settings
       dataset = h5_file.create_dataset(
           dataset_name,
           shape=data_shape,
           chunks=chunk_shape,
           compression='gzip',
           compression_opts=6,
           shuffle=True,
           fletcher32=True,
           scaleoffset=2 if data_shape[-1] > 100 else None
       )
       
       return dataset

**Parallel Writing**

.. code-block:: python

   def parallel_data_writing(data_batches, h5_out_path, n_workers=4):
       """Write data using parallel workers (requires careful HDF5 handling)."""
       
       from multiprocessing import Pool
       import tempfile
       import os
       
       # Create temporary files for each worker
       temp_files = []
       for i in range(n_workers):
           temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
           temp_files.append(temp_file.name)
           temp_file.close()
       
       # Distribute batches among workers
       batch_chunks = [data_batches[i::n_workers] for i in range(n_workers)]
       
       def write_worker_data(args):
           worker_id, worker_batches, temp_path = args
           
           with h5py.File(temp_path, 'w') as h5_temp:
               writer = RawDataWriter(h5_temp)
               
               for batch_idx, batch_data in enumerate(worker_batches):
                   start_idx = batch_idx * len(batch_data['refcodes'])
                   writer.write_raw_crystal_data(start_idx, batch_data['raw_crystal'])
           
           return worker_id, temp_path
       
       # Execute parallel writing
       with Pool(n_workers) as pool:
           worker_args = [(i, batch_chunks[i], temp_files[i]) for i in range(n_workers)]
           results = pool.map(write_worker_data, worker_args)
       
       # Merge temporary files into final output
       merge_hdf5_files([result[1] for result in results], h5_out_path)
       
       # Clean up temporary files
       for temp_file in temp_files:
           os.unlink(temp_file)

Cross-References
----------------

**Related CSA Modules:**

* :doc:`../io/data_reader` - Reading data back from written HDF5 files
* :doc:`../io/dataset_initializer` - Initial HDF5 dataset creation and setup
* :doc:`../io/dimension_scanner` - Determining optimal array dimensions for writing
* :doc:`../extraction/structure_post_extraction_processor` - Main user of data writers

**External Dependencies:**

* `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_ - Hierarchical data format
* `h5py <https://www.h5py.org/>`_ - Python interface to HDF5
* `NumPy <https://numpy.org/>`_ - Array operations and data structures
* `PyTorch <https://pytorch.org/>`_ - Tensor operations and GPU computing

**Performance References:**

* HDF5 Performance Guide: https://docs.hdfgroup.org/hdf5/develop/group___h5_p.html
* h5py Performance Tips: https://docs.h5py.org/en/stable/high/dataset.html#reading-writing-data