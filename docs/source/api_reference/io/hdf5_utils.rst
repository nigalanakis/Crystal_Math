hdf5_utils module
=================

.. automodule:: hdf5_utils
   :members:
   :undoc-members:
   :show-inheritance:

HDF5 File Initialization and Management Utilities
--------------------------------------------------

The ``hdf5_utils`` module provides essential utilities for creating, initializing, and managing HDF5 files used throughout the Crystal Structure Analysis pipeline. It handles file creation, group organization, and data storage patterns optimized for crystallographic structure data.

**Key Features:**

* **File initialization** - Create properly structured HDF5 files
* **Group management** - Organize data into logical hierarchies
* **Compression configuration** - Optimize storage efficiency
* **Error handling** - Robust file operations with proper cleanup
* **Metadata management** - Store and retrieve file-level information
* **Cross-platform compatibility** - Consistent behavior across operating systems

Core Functions
--------------

.. autofunction:: hdf5_utils.initialize_hdf5_file

   **Create and Initialize HDF5 File Structure**

   Creates a new HDF5 file or opens an existing one with the required group structure for CSA data storage. Ensures proper organization and compression settings.

   **Parameters:**

   * **hdf5_path** (:obj:`Path`) - Path to the HDF5 file to create or open
   * **compression** (:obj:`str`, default="gzip") - Compression algorithm for datasets
   * **chunk_size** (:obj:`int`, default=64) - Suggested chunk size for dataset creation

   **Returns:**

   * **h5py.File** - Open HDF5 file handle in append mode

   **File Structure Created:**

   .. code-block:: text

      /
      ├── structures/          # Main data group
      │   ├── <refcode1>/     # Individual structure groups
      │   ├── <refcode2>/
      │   └── ...
      ├── metadata/           # File-level metadata (optional)
      └── processing_log/     # Processing history (optional)

   **Usage Examples:**

   .. code-block:: python

      from pathlib import Path
      from hdf5_utils import initialize_hdf5_file

      # Create new HDF5 file for raw data
      hdf5_path = Path("structures_raw.h5")
      h5_file = initialize_hdf5_file(hdf5_path, compression="gzip", chunk_size=32)
      
      # File is ready for data storage
      print("Available groups:", list(h5_file.keys()))
      # Expected: ['structures']
      
      # Close when done
      h5_file.close()

   **Advanced Configuration:**

   .. code-block:: python

      # Configure for large datasets
      large_dataset_file = initialize_hdf5_file(
          Path("large_structures.h5"),
          compression="lzf",  # Faster compression for large files
          chunk_size=128      # Larger chunks for better I/O
      )
      
      # Configure for maximum compression
      compressed_file = initialize_hdf5_file(
          Path("compressed_structures.h5"),
          compression="gzip", 
          chunk_size=16       # Smaller chunks for better compression
      )

   **Error Handling:**

   .. code-block:: python

      try:
          h5_file = initialize_hdf5_file(hdf5_path)
          # Use file...
      except OSError as e:
          print(f"Failed to create HDF5 file: {e}")
          # Handle error appropriately
      finally:
          if 'h5_file' in locals() and h5_file:
              h5_file.close()

.. autofunction:: hdf5_utils.write_structure_group

   **Store Raw Structure Data in HDF5 Groups**

   Writes raw structure data as JSON strings into organized HDF5 groups with proper metadata and compression.

   **Parameters:**

   * **h5** (:obj:`h5py.File`) - Open HDF5 file with '/structures' group
   * **refcode** (:obj:`str`) - CSD refcode identifier for the structure
   * **raw_json** (:obj:`str`) - Serialized JSON string containing raw structure data

   **Group Organization:**

   Creates or updates the structure: ``/structures/<refcode>/raw_json``

   **Data Storage Features:**

   * **Automatic replacement** - Overwrites existing data if present
   * **UTF-8 encoding** - Proper string encoding for international characters
   * **Compression** - Applies file-level compression settings
   * **Metadata preservation** - Maintains data provenance information

   **Usage Examples:**

   .. code-block:: python

      import json
      
      # Prepare structure data
      structure_data = {
          'identifier': 'ALANIN',
          'cell_lengths': [5.784, 12.133, 5.955],
          'cell_angles': [90.0, 90.0, 90.0],
          'space_group': 'P n a 21',
          'atoms': [
              {'label': 'N1', 'symbol': 'N', 'coords': [0.1, 0.2, 0.3]},
              {'label': 'C1', 'symbol': 'C', 'coords': [0.15, 0.25, 0.35]}
          ]
      }
      
      # Serialize to JSON
      raw_json = json.dumps(structure_data, indent=2)
      
      # Write to HDF5
      with initialize_hdf5_file(Path("structures.h5")) as h5_file:
          write_structure_group(h5_file, "ALANIN", raw_json)
          
          # Verify storage
          stored_data = h5_file['/structures/ALANIN/raw_json'][()].decode()
          restored_structure = json.loads(stored_data)
          print(f"Stored structure: {restored_structure['identifier']}")

   **Batch Writing Pattern:**

   .. code-block:: python

      def write_multiple_structures(h5_file, structure_dict):
          """Write multiple structures efficiently."""
          
          for refcode, structure_data in structure_dict.items():
              # Serialize structure data
              raw_json = json.dumps(structure_data, separators=(',', ':'))  # Compact JSON
              
              # Write to HDF5
              write_structure_group(h5_file, refcode, raw_json)
              
              print(f"✓ Stored {refcode}")
          
          print(f"Successfully stored {len(structure_dict)} structures")

   **Error Handling and Validation:**

   .. code-block:: python

      def robust_structure_writing(h5_file, refcode, structure_data):
          """Write structure data with validation and error handling."""
          
          try:
              # Validate refcode
              if not refcode or not isinstance(refcode, str):
                  raise ValueError("Invalid refcode: must be non-empty string")
              
              # Validate structure data
              if not isinstance(structure_data, dict):
                  raise ValueError("Structure data must be a dictionary")
              
              # Check for required fields
              required_fields = ['identifier', 'cell_lengths', 'cell_angles']
              for field in required_fields:
                  if field not in structure_data:
                      raise KeyError(f"Missing required field: {field}")
              
              # Serialize with error handling
              try:
                  raw_json = json.dumps(structure_data, indent=2, ensure_ascii=False)
              except (TypeError, ValueError) as e:
                  raise ValueError(f"Failed to serialize structure data: {e}")
              
              # Write to HDF5
              write_structure_group(h5_file, refcode, raw_json)
              
              return True
              
          except Exception as e:
              print(f"Error writing structure {refcode}: {e}")
              return False

Advanced HDF5 Management
------------------------

**File Metadata Management**

.. code-block:: python

   def add_file_metadata(h5_file, metadata_dict):
       """Add metadata to HDF5 file for provenance tracking."""
       
       import datetime
       
       # Create metadata group if it doesn't exist
       if 'metadata' not in h5_file:
           meta_group = h5_file.create_group('metadata')
       else:
           meta_group = h5_file['metadata']
       
       # Add standard metadata
       meta_group.attrs['creation_date'] = datetime.datetime.now().isoformat()
       meta_group.attrs['csa_version'] = metadata_dict.get('csa_version', 'unknown')
       meta_group.attrs['total_structures'] = metadata_dict.get('total_structures', 0)
       meta_group.attrs['processing_stage'] = metadata_dict.get('stage', 'raw')
       
       # Add custom metadata
       for key, value in metadata_dict.items():
           if key not in ['creation_date', 'csa_version', 'total_structures', 'processing_stage']:
               meta_group.attrs[key] = value
       
       print(f"Added metadata to HDF5 file: {dict(meta_group.attrs)}")

   def read_file_metadata(h5_file):
       """Read metadata from HDF5 file."""
       
       if 'metadata' in h5_file:
           metadata = dict(h5_file['metadata'].attrs)
           return metadata
       else:
           return {}

**Compression Optimization**

.. code-block:: python

   def optimize_hdf5_compression(h5_file_path, compression_level=6):
       """Optimize HDF5 file compression settings."""
       
       import h5py
       
       # Define compression strategies for different data types
       compression_strategies = {
           'float32': {
               'compression': 'gzip',
               'compression_opts': compression_level,
               'shuffle': True,
               'scaleoffset': 2
           },
           'float64': {
               'compression': 'gzip', 
               'compression_opts': compression_level,
               'shuffle': True,
               'scaleoffset': 3
           },
           'int32': {
               'compression': 'gzip',
               'compression_opts': compression_level,
               'shuffle': True
           },
           'string': {
               'compression': 'gzip',
               'compression_opts': compression_level
           }
       }
       
       return compression_strategies

   def create_compressed_dataset(h5_group, name, data, dtype_hint=None):
       """Create optimally compressed dataset."""
       
       import numpy as np
       
       # Determine data type
       if dtype_hint:
           dtype_str = dtype_hint
       elif hasattr(data, 'dtype'):
           dtype_str = str(data.dtype)
       else:
           dtype_str = 'string'
       
       # Get compression settings
       compression_opts = optimize_hdf5_compression(None)[dtype_str]
       
       # Create dataset with optimal settings
       if isinstance(data, str):
           # Handle string data
           dataset = h5_group.create_dataset(
               name,
               data=data,
               dtype=h5py.string_dtype(encoding='utf-8'),
               **{k: v for k, v in compression_opts.items() if k != 'scaleoffset'}
           )
       else:
           # Handle numeric data
           dataset = h5_group.create_dataset(
               name,
               data=data,
               **compression_opts
           )
       
       return dataset

**File Integrity and Validation**

.. code-block:: python

   def validate_hdf5_file(h5_file_path):
       """Validate HDF5 file structure and integrity."""
       
       validation_report = {
           'is_valid': True,
           'errors': [],
           'warnings': [],
           'structure_count': 0,
           'groups': [],
           'file_size_mb': 0
       }
       
       try:
           with h5py.File(h5_file_path, 'r') as h5_file:
               # Check file size
               import os
               file_size = os.path.getsize(h5_file_path) / (1024 * 1024)
               validation_report['file_size_mb'] = round(file_size, 2)
               
               # Check required groups
               if 'structures' not in h5_file:
                   validation_report['errors'].append("Missing required 'structures' group")
                   validation_report['is_valid'] = False
               else:
                   structures_group = h5_file['structures']
                   structure_keys = list(structures_group.keys())
                   validation_report['structure_count'] = len(structure_keys)
                   validation_report['groups'].extend(structure_keys[:10])  # First 10
                   
                   # Check individual structures
                   for refcode in structure_keys[:10]:  # Sample validation
                       structure_group = structures_group[refcode]
                       
                       if 'raw_json' not in structure_group:
                           validation_report['warnings'].append(f"Structure {refcode} missing raw_json")
                       else:
                           try:
                               # Try to decode JSON
                               raw_data = structure_group['raw_json'][()].decode()
                               json.loads(raw_data)
                           except Exception as e:
                               validation_report['errors'].append(f"Invalid JSON in {refcode}: {e}")
                               validation_report['is_valid'] = False
               
               # Check metadata
               if 'metadata' in h5_file:
                   metadata = dict(h5_file['metadata'].attrs)
                   if 'creation_date' not in metadata:
                       validation_report['warnings'].append("Missing creation_date in metadata")
       
       except Exception as e:
           validation_report['errors'].append(f"Failed to open file: {e}")
           validation_report['is_valid'] = False
       
       return validation_report

   def repair_hdf5_file(h5_file_path, backup=True):
       """Attempt to repair corrupted HDF5 file."""
       
       if backup:
           import shutil
           backup_path = h5_file_path.with_suffix('.h5.backup')
           shutil.copy2(h5_file_path, backup_path)
           print(f"Created backup: {backup_path}")
       
       try:
           # Try to open and repack the file
           with h5py.File(h5_file_path, 'r') as h5_in:
               temp_path = h5_file_path.with_suffix('.h5.temp')
               
               with h5py.File(temp_path, 'w') as h5_out:
                   # Copy all accessible data
                   def copy_item(name, obj):
                       try:
                           h5_in.copy(obj, h5_out, name)
                           return True
                       except Exception as e:
                           print(f"Failed to copy {name}: {e}")
                           return False
                   
                   h5_in.visititems(copy_item)
           
           # Replace original with repaired version
           h5_file_path.unlink()
           temp_path.rename(h5_file_path)
           
           print(f"Successfully repaired {h5_file_path}")
           return True
           
       except Exception as e:
           print(f"Failed to repair file: {e}")
           return False

**File Organization Utilities**

.. code-block:: python

   def organize_hdf5_structure(h5_file, organization_scheme='by_space_group'):
       """Reorganize HDF5 file structure for better access patterns."""
       
       if organization_scheme == 'by_space_group':
           organize_by_space_group(h5_file)
       elif organization_scheme == 'by_composition':
           organize_by_composition(h5_file)
       elif organization_scheme == 'by_size':
           organize_by_size(h5_file)
       else:
           raise ValueError(f"Unknown organization scheme: {organization_scheme}")

   def organize_by_space_group(h5_file):
       """Reorganize structures by space group."""
       
       if 'structures' not in h5_file:
           return
       
       structures = h5_file['structures']
       space_group_map = {}
       
       # Collect structures by space group
       for refcode in structures.keys():
           try:
               raw_data = structures[refcode]['raw_json'][()].decode()
               structure_data = json.loads(raw_data)
               space_group = structure_data.get('space_group', 'unknown')
               
               if space_group not in space_group_map:
                   space_group_map[space_group] = []
               space_group_map[space_group].append(refcode)
               
           except Exception as e:
               print(f"Warning: Could not process {refcode}: {e}")
       
       # Create space group organization
       if 'by_space_group' not in h5_file:
           sg_group = h5_file.create_group('by_space_group')
       else:
           sg_group = h5_file['by_space_group']
       
       for space_group, refcodes in space_group_map.items():
           sg_subgroup = sg_group.create_group(space_group)
           sg_subgroup.attrs['structure_count'] = len(refcodes)
           sg_subgroup.create_dataset('refcodes', data=refcodes, dtype=h5py.string_dtype())
       
       print(f"Organized {len(structures)} structures into {len(space_group_map)} space groups")

Performance Optimization
------------------------

**Chunking Strategies**

.. code-block:: python

   def calculate_optimal_chunks(data_shape, target_chunk_size_mb=1.0):
       """Calculate optimal chunk sizes for HDF5 datasets."""
       
       import numpy as np
       
       # Estimate bytes per element
       bytes_per_element = 4  # Assume float32
       total_elements = np.prod(data_shape)
       total_size_mb = (total_elements * bytes_per_element) / (1024 * 1024)
       
       if total_size_mb <= target_chunk_size_mb:
           # Small dataset - no chunking needed
           return None
       
       # Calculate chunk dimensions
       if len(data_shape) == 1:
           chunk_elements = int(target_chunk_size_mb * 1024 * 1024 / bytes_per_element)
           return (min(chunk_elements, data_shape[0]),)
       
       elif len(data_shape) == 2:
           # Chunk by rows
           rows_per_chunk = max(1, int(target_chunk_size_mb * 1024 * 1024 / (data_shape[1] * bytes_per_element)))
           return (min(rows_per_chunk, data_shape[0]), data_shape[1])
       
       elif len(data_shape) == 3:
           # Chunk by 2D slices
           return (1, data_shape[1], data_shape[2])
       
       else:
           # Multi-dimensional - chunk first dimension
           return (1,) + data_shape[1:]

   def create_optimized_dataset(h5_group, name, data, chunk_size_mb=1.0):
       """Create dataset with optimal chunking and compression."""
       
       # Calculate optimal chunks
       chunks = calculate_optimal_chunks(data.shape, chunk_size_mb)
       
       # Create dataset with optimization
       dataset = h5_group.create_dataset(
           name,
           data=data,
           chunks=chunks,
           compression='gzip',
           compression_opts=6,
           shuffle=True,
           fletcher32=True
       )
       
       return dataset

**I/O Performance Monitoring**

.. code-block:: python

   def monitor_hdf5_performance(func):
       """Decorator to monitor HDF5 I/O performance."""
       
       import time
       import functools
       
       @functools.wraps(func)
       def wrapper(*args, **kwargs):
           start_time = time.time()
           
           result = func(*args, **kwargs)
           
           end_time = time.time()
           duration = end_time - start_time
           
           print(f"{func.__name__} completed in {duration:.2f} seconds")
           
           return result
       
       return wrapper

   @monitor_hdf5_performance
   def write_large_dataset(h5_file, dataset_name, data):
       """Write large dataset with performance monitoring."""
       
       dataset = create_optimized_dataset(h5_file, dataset_name, data)
       return dataset

**Memory Management**

.. code-block:: python

   def memory_efficient_hdf5_copy(source_path, dest_path, chunk_size=1000):
       """Copy HDF5 file in chunks to manage memory usage."""
       
       with h5py.File(source_path, 'r') as h5_source:
           with h5py.File(dest_path, 'w') as h5_dest:
               
               def copy_dataset_chunked(source_dataset, dest_group, name):
                   """Copy dataset in chunks."""
                   
                   # Create destination dataset
                   dest_dataset = dest_group.create_dataset(
                       name,
                       shape=source_dataset.shape,
                       dtype=source_dataset.dtype,
                       chunks=True,
                       compression='gzip'
                   )
                   
                   # Copy data in chunks
                   if len(source_dataset.shape) == 1:
                       for start in range(0, source_dataset.shape[0], chunk_size):
                           end = min(start + chunk_size, source_dataset.shape[0])
                           dest_dataset[start:end] = source_dataset[start:end]
                   
                   elif len(source_dataset.shape) == 2:
                       for start in range(0, source_dataset.shape[0], chunk_size):
                           end = min(start + chunk_size, source_dataset.shape[0])
                           dest_dataset[start:end, :] = source_dataset[start:end, :]
                   
                   # Copy attributes
                   for attr_name, attr_value in source_dataset.attrs.items():
                       dest_dataset.attrs[attr_name] = attr_value
               
               # Copy all groups and datasets
               def copy_item(name, obj):
                   if isinstance(obj, h5py.Group):
                       h5_dest.create_group(name)
                   elif isinstance(obj, h5py.Dataset):
                       group_name = '/'.join(name.split('/')[:-1])
                       dataset_name = name.split('/')[-1]
                       
                       if group_name:
                           dest_group = h5_dest[group_name]
                       else:
                           dest_group = h5_dest
                       
                       copy_dataset_chunked(obj, dest_group, dataset_name)
               
               h5_source.visititems(copy_item)

Integration Examples
--------------------

**Complete File Management Pipeline**

.. code-block:: python

   def create_and_populate_hdf5_file(structures_data, output_path):
       """Complete pipeline for creating and populating HDF5 file."""
       
       # Initialize file
       h5_file = initialize_hdf5_file(output_path, compression="gzip")
       
       try:
           # Add metadata
           metadata = {
               'csa_version': '1.0.0',
               'total_structures': len(structures_data),
               'stage': 'raw_extraction',
               'data_source': 'CSD_2024'
           }
           add_file_metadata(h5_file, metadata)
           
           # Write structures
           successful_writes = 0
           failed_writes = []
           
           for refcode, structure_data in structures_data.items():
               try:
                   raw_json = json.dumps(structure_data, indent=2)
                   write_structure_group(h5_file, refcode, raw_json)
                   successful_writes += 1
               except Exception as e:
                   failed_writes.append((refcode, str(e)))
           
           # Update metadata with results
           h5_file['metadata'].attrs['successful_writes'] = successful_writes
           h5_file['metadata'].attrs['failed_writes'] = len(failed_writes)
           
           print(f"Successfully wrote {successful_writes} structures")
           if failed_writes:
               print(f"Failed to write {len(failed_writes)} structures:")
               for refcode, error in failed_writes[:5]:  # Show first 5 errors
                   print(f"  {refcode}: {error}")
           
           return h5_file
           
       except Exception as e:
           h5_file.close()
           raise e

   def validate_and_repair_if_needed(h5_file_path):
       """Validate HDF5 file and repair if necessary."""
       
       # First, validate the file
       validation_report = validate_hdf5_file(h5_file_path)
       
       print("Validation Report:")
       print(f"  Valid: {validation_report['is_valid']}")
       print(f"  Structures: {validation_report['structure_count']}")
       print(f"  File size: {validation_report['file_size_mb']} MB")
       
       if validation_report['errors']:
           print("  Errors:")
           for error in validation_report['errors']:
               print(f"    - {error}")
       
       if validation_report['warnings']:
           print("  Warnings:")
           for warning in validation_report['warnings']:
               print(f"    - {warning}")
       
       # Attempt repair if needed
       if not validation_report['is_valid']:
           print("\nAttempting to repair file...")
           if repair_hdf5_file(h5_file_path):
               # Re-validate after repair
               new_validation = validate_hdf5_file(h5_file_path)
               if new_validation['is_valid']:
                   print("File successfully repaired and validated")
               else:
                   print("Repair attempt failed")
           else:
               print("Could not repair file")
       
       return validation_report['is_valid']

Cross-References
----------------

**Related CSA Modules:**

* :doc:`../io/data_reader` - Reading data from initialized HDF5 files
* :doc:`../io/data_writer` - Writing structured data to HDF5 files
* :doc:`../io/dataset_initializer` - Creating dataset schemas in HDF5 files
* :doc:`../extraction/structure_data_extractor` - Primary user of HDF5 utilities

**External Dependencies:**

* `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_ - Hierarchical data format library
* `h5py <https://www.h5py.org/>`_ - Python interface to HDF5
* `pathlib` - Path manipulation utilities

**File Format Documentation:**

* HDF5 User Guide: https://docs.hdfgroup.org/hdf5/develop/_u_g.html
* HDF5 Best Practices: https://docs.hdfgroup.org/hdf5/develop/_best_practices.html
* h5py Advanced Features: https://docs.h5py.org/en/stable/high/group.html