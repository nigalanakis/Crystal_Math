structure_data_extractor module
===============================

.. automodule:: structure_data_extractor
   :members:
   :undoc-members:
   :show-inheritance:

Raw Structure Data Extraction Pipeline
---------------------------------------

The ``structure_data_extractor`` module manages parallel extraction of comprehensive structural data from the Cambridge Structural Database into optimized HDF5 containers. This module forms the core of Stage 4 in the CSA pipeline.

StructureDataExtractor Class
----------------------------

.. autoclass:: structure_data_extractor.StructureDataExtractor
   :members:
   :undoc-members:
   :show-inheritance:

   **Parallel HDF5 Data Extraction Manager**

   Orchestrates the extraction of raw structural data from CSD entries into a structured HDF5 format optimized for downstream processing and analysis.

   **Core Functionality:**

   * **Parallel Processing** - Distributes extraction across multiple CPU cores
   * **HDF5 Management** - Creates and manages structured datasets with proper typing
   * **Data Validation** - Ensures data integrity and completeness
   * **Memory Optimization** - Processes structures in configurable batch sizes
   * **Error Handling** - Robust error recovery and logging

   **Data Extracted:**

   **Crystal-Level Data:**
   * Space group symmetry and operators
   * Unit cell parameters (lengths, angles, volume)
   * Z and Z' values
   * Density and packing coefficient

   **Molecular Data:**
   * Atomic coordinates (Cartesian and fractional)
   * Atomic properties (element, charge, formal charge)
   * Bond connectivity and bond orders
   * Molecular topology and geometry

   **Intermolecular Interactions:**
   * Close contact identification and distances
   * Hydrogen bond detection and geometry
   * Symmetry-expanded contact networks
   * Contact classification and properties

   **Attributes:**
      * **hdf5_path** (:obj:`Path`) - Output HDF5 file location
      * **filters** (:obj:`Dict[str, Any]`) - Extraction configuration and filters
      * **batch_size** (:obj:`int`) - Structures processed per batch
      * **reader** (:obj:`io.EntryReader`) - CCDC database connection

   .. automethod:: __init__

      **Initialize Structure Data Extractor**

      Parameters:
         * **hdf5_path** (:obj:`Union[str, Path]`) - Path for output HDF5 file
         * **filters** (:obj:`Dict[str, Any]`) - Extraction configuration dictionary
         * **batch_size** (:obj:`int`) - Number of structures per processing batch

      **Filter Configuration:**

      The filters dictionary contains:

      .. code-block:: python

         filters = {
             "data_directory": "./output",      # Base output directory
             "data_prefix": "analysis",         # Filename prefix
             "center_molecule": False,          # Center molecule in unit cell
             "target_species": ["C","H","N","O"], # Allowed elements
             "molecule_weight_limit": 500.0,    # Molecular weight limit
             "target_z_prime_values": [1],      # Z' constraints
             # ... additional validation criteria
         }

      **Initialization Process:**

      .. code-block:: python

         # Set up file paths
         self.hdf5_path = Path(hdf5_path)
         self.filters = filters
         self.batch_size = batch_size
         
         # Initialize CSD connection
         self.reader = io.EntryReader("CSD")

   .. automethod:: run

      **Execute Complete Data Extraction Pipeline**

      Orchestrates the full extraction workflow from refcode list to populated HDF5 file.

      **Pipeline Stages:**

      1. **File Initialization** - Remove existing HDF5 and create new structure
      2. **Refcode Loading** - Read list of structures to extract
      3. **HDF5 Setup** - Initialize datasets and group structure
      4. **Batch Processing** - Extract data in parallel batches
      5. **Data Writing** - Store structured data with proper typing
      6. **Validation** - Verify data integrity and completeness

      **HDF5 Structure Created:**

      .. code-block:: text

         analysis.h5
         ├── /refcode_list              # List of all extracted refcodes
         └── /structures/               # Root group for structure data
             ├── REFCODE01/             # Individual structure group
             │   ├── identifier         # CSD refcode
             │   ├── space_group        # Space group symbol
             │   ├── z_value           # Z value
             │   ├── z_prime           # Z' value
             │   ├── cell_volume       # Unit cell volume
             │   ├── cell_density      # Crystal density
             │   ├── packing_coefficient # Packing coefficient
             │   ├── cell_lengths      # a, b, c parameters
             │   ├── cell_angles       # α, β, γ angles
             │   ├── symmetry_operators # Space group operators
             │   ├── atom_labels       # Atomic labels
             │   ├── atom_coordinates  # Cartesian coordinates
             │   ├── atom_frac_coordinates # Fractional coordinates
             │   ├── atom_atomic_weights # Atomic masses
             │   ├── atom_atomic_numbers # Atomic numbers
             │   ├── atom_formal_charges # Formal charges
             │   ├── atom_symbols      # Element symbols
             │   ├── atom_sybyl_types  # SYBYL atom types
             │   ├── atom_neighbours   # Connectivity lists
             │   ├── bond_atom1_labels # Bond partner 1
             │   ├── bond_atom2_labels # Bond partner 2
             │   ├── bond_orders       # Bond order values
             │   ├── intermolecular_contacts/ # Contact data
             │   └── intermolecular_hbonds/   # H-bond data
             └── REFCODE02/             # Next structure
                 └── ...

      **Progress Monitoring:**

      .. code-block:: text

         INFO - Initializing HDF5 file: analysis.h5
         INFO - Loaded 8234 refcodes for extraction
         INFO - Processing batch 1/258 (32 structures)
         INFO - Processing batch 2/258 (32 structures)
         ...
         INFO - Extraction complete: 8234 structures processed
         INFO - HDF5 file closed: analysis.h5

      **Error Handling:**

      * **Structure Failures** - Individual structure errors are logged and skipped
      * **Batch Failures** - Failed batches are retried with smaller sizes
      * **HDF5 Errors** - File corruption triggers automatic recovery
      * **Memory Issues** - Out-of-memory conditions reduce batch sizes

      **Raises:**
         * :obj:`FileNotFoundError` - If refcode list CSV is missing
         * :obj:`Exception` - For CCDC API or HDF5 operation failures

   .. automethod:: _load_refcodes

      **Load Refcode List from CSV**

      Reads the list of structures to extract from the unique structures CSV file.

      **File Sources:**

      The method searches for refcode lists in order of preference:

      1. **Unique structures** - ``{prefix}_refcode_families_unique.csv``
      2. **Clustered families** - ``{prefix}_refcode_families_clustered.csv``
      3. **All families** - ``{prefix}_refcode_families.csv``

      **CSV Format Expected:**

      .. code-block:: text

         family_id,refcode
         FAM001,ABINIK01
         FAM002,ACETAC
         FAM003,BENZAC02
         ...

      **Returns:**
         :obj:`List[str]` - List of CSD refcodes to extract

      **Error Handling:**

      .. code-block:: python

         # Multiple fallback options
         csv_files = [
             f"{self.filters['data_prefix']}_refcode_families_unique.csv",
             f"{self.filters['data_prefix']}_refcode_families_clustered.csv", 
             f"{self.filters['data_prefix']}_refcode_families.csv"
         ]
         
         for csv_file in csv_files:
             csv_path = Path(self.filters['data_directory']) / csv_file
             if csv_path.exists():
                 return pd.read_csv(csv_path)['refcode'].tolist()
         
         raise FileNotFoundError("No refcode CSV files found")

   .. automethod:: _process_batch

      **Process Structure Batch with Parallel Extraction**

      Extracts raw data for a batch of structures using parallel processing and writes results to HDF5.

      **Parameters:**
         * **batch** (:obj:`List[str]`) - Refcodes to process in this batch
         * **h5** (:obj:`h5py.File`) - Open HDF5 file handle

      **Parallel Processing:**

      .. code-block:: python

         # Optimize worker count
         max_workers = min(len(batch), (os.cpu_count() or 1) - 1)
         
         # Submit extraction jobs
         with ProcessPoolExecutor(max_workers=max_workers) as executor:
             futures = {
                 executor.submit(_extract_one, refcode, self.filters, center_flag): refcode
                 for refcode in batch
             }
             
             # Collect results
             for future in as_completed(futures):
                 result = future.result()
                 if result:
                     results.append(result)

      **Data Writing Process:**

      For each structure, creates properly typed HDF5 datasets:

      .. code-block:: python

         # Create structure group
         grp = h5["structures"].require_group(refcode)
         
         # Write crystal data with appropriate types
         grp.create_dataset("space_group", data=space_group, 
                           dtype=h5py.string_dtype(encoding="utf-8"))
         grp.create_dataset("cell_volume", data=volume, dtype=np.float32)
         grp.create_dataset("atom_coordinates", data=coords, dtype=np.float32)
         
         # Handle variable-length data
         grp.create_dataset("atom_neighbours", data=neighbors, 
                           dtype=h5py.string_dtype(encoding="utf-8"))

      **Memory Management:**

      * **Batch-wise processing** prevents memory overflow
      * **Immediate writing** avoids accumulating data in memory
      * **Garbage collection** triggered between batches
      * **Resource monitoring** tracks memory usage

      **Error Recovery:**

      .. code-block:: python

         try:
             # Process batch
             results = extract_batch_parallel(batch)
         except MemoryError:
             # Reduce batch size and retry
             smaller_batches = split_batch(batch, size=batch_size//2)
             results = []
             for small_batch in smaller_batches:
                 results.extend(extract_batch_parallel(small_batch))

Data Extraction Functions
-------------------------

**_extract_one**

.. autofunction:: structure_data_extractor._extract_one

   **Extract Complete Data for Single Structure**

   Worker function that extracts all structural data for a single CSD entry.

   **Parameters:**
      * **refcode** (:obj:`str`) - CSD refcode to extract
      * **filters** (:obj:`Dict[str, Any]`) - Extraction configuration
      * **center_molecule** (:obj:`bool`) - Whether to center molecule in unit cell

   **Extraction Process:**

   1. **CSD Entry Loading** - Connect to database and load entry
   2. **Crystal Data** - Extract unit cell and symmetry information
   3. **Molecular Data** - Extract atomic coordinates and connectivity
   4. **Contact Analysis** - Identify intermolecular interactions
   5. **Data Packaging** - Organize data for HDF5 storage

   **Crystal Data Extracted:**

   .. code-block:: python

      crystal_data = {
          "identifier": entry.identifier,
          "space_group": entry.crystal.spacegroup_symbol,
          "z_value": entry.crystal.z_value,
          "z_prime": entry.crystal.z_prime,
          "cell_volume": entry.crystal.cell_volume,
          "cell_density": entry.crystal.calculated_density,
          "packing_coefficient": entry.crystal.packing_coefficient,
          "cell_lengths": [a, b, c],
          "cell_angles": [alpha, beta, gamma],
          "symmetry_operators": symmetry_ops
      }

   **Molecular Data Extracted:**

   .. code-block:: python

      atom_data = {
          "coordinates": cartesian_coords,
          "fractional_coordinates": fractional_coords,
          "atomic_weights": atomic_masses,
          "atomic_numbers": atomic_numbers,
          "formal_charges": formal_charges,
          "symbols": element_symbols,
          "sybyl_types": sybyl_atom_types,
          "neighbours": connectivity_lists
      }

   **Contact Data Extracted:**

   .. code-block:: python

      contact_data = {
          "central_atom": contact.atoms[0].label,
          "contact_atom": contact.atoms[1].label,
          "distance": contact.distance,
          "is_intermolecular": contact.intermolecular,
          "symmetry_operator": contact.symmetry_operator
      }

   **Hydrogen Bond Data:**

   .. code-block:: python

      hbond_data = {
          "donor_atom": hbond.atoms[0].label,
          "hydrogen_atom": hbond.atoms[1].label,
          "acceptor_atom": hbond.atoms[2].label,
          "da_distance": hbond.da_distance,
          "angle": hbond.angle,
          "is_intermolecular": hbond.intermolecular,
          "is_in_line_of_sight": hbond.is_in_line_of_sight
      }

   **Returns:**
      :obj:`Tuple[str, Dict[str, Any]]` - (refcode, extracted_data)

   **Error Handling:**

   * **Missing structures** - Gracefully handle unavailable refcodes
   * **Corrupted data** - Skip structures with data integrity issues
   * **API failures** - Retry with exponential backoff
   * **Timeout handling** - Abort long-running extractions

HDF5 Data Organization
----------------------

**File Structure Design**

The HDF5 file is organized for optimal access patterns:

.. code-block:: text

   analysis.h5
   ├── /refcode_list                    # 1D string array
   ├── /structures/                     # Group
   │   ├── REFCODE01/                   # Structure group
   │   │   ├── identifier               # String scalar
   │   │   ├── space_group              # String scalar
   │   │   ├── z_value                  # Integer scalar
   │   │   ├── z_prime                  # Float scalar
   │   │   ├── cell_volume              # Float scalar
   │   │   ├── cell_density             # Float scalar
   │   │   ├── packing_coefficient      # Float scalar
   │   │   ├── cell_lengths             # Float array (3,)
   │   │   ├── cell_angles              # Float array (3,)
   │   │   ├── symmetry_operators       # String array (N,)
   │   │   ├── atom_labels              # String array (M,)
   │   │   ├── atom_coordinates         # Float array (M, 3)
   │   │   ├── atom_frac_coordinates    # Float array (M, 3)
   │   │   ├── atom_atomic_weights      # Float array (M,)
   │   │   ├── atom_atomic_numbers      # Integer array (M,)
   │   │   ├── atom_formal_charges      # Float array (M,)
   │   │   ├── atom_symbols             # String array (M,)
   │   │   ├── atom_sybyl_types         # String array (M,)
   │   │   ├── atom_neighbours          # Variable-length strings
   │   │   ├── bond_atom1_labels        # String array (B,)
   │   │   ├── bond_atom2_labels        # String array (B,)
   │   │   ├── bond_orders              # Float array (B,)
   │   │   ├── intermolecular_contacts/ # Group
   │   │   │   ├── central_atoms        # String array (C,)
   │   │   │   ├── contact_atoms        # String array (C,)
   │   │   │   ├── distances            # Float array (C,)
   │   │   │   └── symmetry_operators   # String array (C,)
   │   │   └── intermolecular_hbonds/   # Group
   │   │       ├── donor_atoms          # String array (H,)
   │   │       ├── hydrogen_atoms       # String array (H,)
   │   │       ├── acceptor_atoms       # String array (H,)
   │   │       ├── da_distances         # Float array (H,)
   │   │       ├── angles               # Float array (H,)
   │   │       └── line_of_sight        # Boolean array (H,)
   │   └── REFCODE02/
   │       └── ...

**Data Type Optimization**

Careful attention to data types minimizes file size:

.. code-block:: python

   # Use appropriate precision for coordinates
   coordinates = np.array(coords, dtype=np.float32)  # 32-bit sufficient
   
   # Integer types for discrete values
   atomic_numbers = np.array(numbers, dtype=np.int16)  # 16-bit sufficient
   
   # String types with UTF-8 encoding
   labels = np.array(labels, dtype=h5py.string_dtype(encoding="utf-8"))
   
   # Boolean arrays for flags
   intermolecular_flags = np.array(flags, dtype=bool)

**Compression and Chunking**

.. code-block:: python

   # Apply compression for large datasets
   grp.create_dataset(
       "atom_coordinates", 
       data=coordinates,
       dtype=np.float32,
       compression="gzip",
       compression_opts=6,
       shuffle=True,
       fletcher32=True
   )

Usage Examples
--------------

**Basic Extraction Workflow**

.. code-block:: python

   from structure_data_extractor import StructureDataExtractor
   from pathlib import Path

   # Configure extraction
   extractor = StructureDataExtractor(
       hdf5_path=Path("./output/analysis.h5"),
       filters={
           "data_directory": "./output",
           "data_prefix": "analysis",
           "center_molecule": False,
           "target_species": ["C", "H", "N", "O"]
       },
       batch_size=32
   )

   # Execute extraction
   extractor.run()

**Memory-Optimized Extraction**

.. code-block:: python

   # Configure for limited memory systems
   memory_limited_extractor = StructureDataExtractor(
       hdf5_path="./analysis_small.h5",
       filters=standard_filters,
       batch_size=8  # Smaller batches for memory efficiency
   )

   # Monitor memory usage during extraction
   import psutil
   import logging

   def monitor_extraction():
       process = psutil.Process()
       initial_memory = process.memory_info().rss / 1024**2  # MB
       
       logging.info(f"Initial memory usage: {initial_memory:.1f} MB")
       
       memory_limited_extractor.run()
       
       final_memory = process.memory_info().rss / 1024**2
       logging.info(f"Final memory usage: {final_memory:.1f} MB")
       logging.info(f"Memory increase: {final_memory - initial_memory:.1f} MB")

**High-Throughput Extraction**

.. code-block:: python

   # Configure for maximum throughput
   high_throughput_extractor = StructureDataExtractor(
       hdf5_path="./large_analysis.h5",
       filters=permissive_filters,
       batch_size=128  # Large batches for throughput
   )

   # Use SSD storage for temporary files
   import tempfile
   import shutil

   with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
       temp_path = Path(temp_dir) / "temp_analysis.h5"
       
       # Extract to fast temporary storage
       temp_extractor = StructureDataExtractor(
           hdf5_path=temp_path,
           filters=filters,
           batch_size=128
       )
       temp_extractor.run()
       
       # Move to permanent storage
       shutil.move(temp_path, "./permanent/analysis.h5")

**Custom Data Filtering**

.. code-block:: python

   # Pharmaceutical-specific extraction
   pharma_filters = {
       "data_directory": "./pharma_analysis",
       "data_prefix": "drug_crystals",
       "center_molecule": True,
       "target_species": ["C", "H", "N", "O", "S", "Cl", "F"],
       "molecule_weight_limit": 800.0,
       "target_z_prime_values": [1, 2],
       "exclude_solvates": False,  # Include solvates for drug analysis
       "min_resolution": 1.5,
       "max_r_factor": 0.05
   }

   pharma_extractor = StructureDataExtractor(
       hdf5_path="./pharma_crystals.h5",
       filters=pharma_filters,
       batch_size=64
   )

**Error Recovery and Monitoring**

.. code-block:: python

   import logging
   import time
   from pathlib import Path

   def robust_extraction(hdf5_path, filters, max_retries=3):
       """Execute extraction with automatic retry on failure."""
       
       for attempt in range(max_retries):
           try:
               extractor = StructureDataExtractor(
                   hdf5_path=hdf5_path,
                   filters=filters,
                   batch_size=32
               )
               
               start_time = time.time()
               extractor.run()
               duration = time.time() - start_time
               
               logging.info(f"Extraction completed in {duration:.1f} seconds")
               return True
               
           except Exception as e:
               logging.error(f"Extraction attempt {attempt + 1} failed: {e}")
               
               # Clean up partial file
               if Path(hdf5_path).exists():
                   Path(hdf5_path).unlink()
               
               if attempt < max_retries - 1:
                   logging.info(f"Retrying in 30 seconds...")
                   time.sleep(30)
               else:
                   logging.error("All extraction attempts failed")
                   raise

**Custom Processing Pipeline**

.. code-block:: python

   from concurrent.futures import ProcessPoolExecutor
   import h5py

   def custom_extraction_pipeline(refcode_lists, output_dir):
       """Process multiple datasets in parallel."""
       
       def extract_dataset(refcodes, output_name):
           extractor = StructureDataExtractor(
               hdf5_path=output_dir / f"{output_name}.h5",
               filters={
                   "data_directory": str(output_dir),
                   "data_prefix": output_name,
                   "center_molecule": False
               },
               batch_size=32
           )
           
           # Temporarily write refcode list
           temp_csv = output_dir / f"{output_name}_temp_refcodes.csv"
           pd.DataFrame({"refcode": refcodes}).to_csv(temp_csv, index=False)
           
           extractor.run()
           
           # Clean up temporary file
           temp_csv.unlink()
           
           return output_name
       
       # Process datasets in parallel
       with ProcessPoolExecutor(max_workers=4) as executor:
           futures = {
               executor.submit(extract_dataset, refcodes, name): name
               for name, refcodes in refcode_lists.items()
           }
           
           for future in futures:
               result = future.result()
               logging.info(f"Completed extraction for {result}")

Data Access Patterns
--------------------

**Reading Extracted Data**

.. code-block:: python

   import h5py
   import numpy as np

   def load_structure_data(hdf5_path, refcode):
       """Load all data for a specific structure."""
       
       with h5py.File(hdf5_path, 'r') as f:
           grp = f[f"structures/{refcode}"]
           
           structure_data = {
               "identifier": grp["identifier"][()].decode(),
               "space_group": grp["space_group"][()].decode(),
               "cell_volume": grp["cell_volume"][()],
               "atom_coordinates": grp["atom_coordinates"][:],
               "atom_symbols": [s.decode() for s in grp["atom_symbols"][:]],
               # ... load other datasets as needed
           }
       
       return structure_data

   def load_all_refcodes(hdf5_path):
       """Get list of all extracted structures."""
       
       with h5py.File(hdf5_path, 'r') as f:
           refcodes = [s.decode() for s in f["refcode_list"][:]]
       
       return refcodes

   def load_crystal_properties(hdf5_path):
       """Load crystal properties for all structures."""
       
       properties = []
       with h5py.File(hdf5_path, 'r') as f:
           for refcode in f["structures"].keys():
               grp = f[f"structures/{refcode}"]
               properties.append({
                   "refcode": refcode,
                   "space_group": grp["space_group"][()].decode(),
                   "volume": grp["cell_volume"][()],
                   "density": grp["cell_density"][()],
                   "z_prime": grp["z_prime"][()]
               })
       
       return pd.DataFrame(properties)

**Batch Data Loading**

.. code-block:: python

   def load_coordinates_batch(hdf5_path, refcodes):
       """Load atomic coordinates for multiple structures."""
       
       coordinates = {}
       with h5py.File(hdf5_path, 'r') as f:
           for refcode in refcodes:
               if f"structures/{refcode}" in f:
                   coords = f[f"structures/{refcode}/atom_coordinates"][:]
                   coordinates[refcode] = coords
       
       return coordinates

   def scan_dataset_sizes(hdf5_path):
       """Analyze dataset sizes for memory planning."""
       
       sizes = []
       with h5py.File(hdf5_path, 'r') as f:
           for refcode in f["structures"].keys():
               grp = f[f"structures/{refcode}"]
               n_atoms = len(grp["atom_coordinates"])
               n_contacts = len(grp["intermolecular_contacts/distances"]) if "intermolecular_contacts/distances" in grp else 0
               
               sizes.append({
                   "refcode": refcode,
                   "n_atoms": n_atoms,
                   "n_contacts": n_contacts
               })
       
       return pd.DataFrame(sizes)

Performance Considerations
--------------------------

**Batch Size Optimization**

.. code-block:: python

   import psutil

   def optimize_batch_size(total_structures, available_memory_gb=None):
       """Determine optimal batch size based on system resources."""
       
       if available_memory_gb is None:
           available_memory_gb = psutil.virtual_memory().available / (1024**3)
       
       # Estimate memory per structure (rough approximation)
       memory_per_structure_mb = 50  # Average for typical organic structures
       
       # Calculate safe batch size
       safe_batch_size = int((available_memory_gb * 1024 * 0.5) / memory_per_structure_mb)
       
       # Apply reasonable bounds
       optimal_batch_size = max(8, min(safe_batch_size, 128))
       
       logging.info(f"Recommended batch size: {optimal_batch_size}")
       return optimal_batch_size

**Storage Optimization**

.. code-block:: python

   def estimate_storage_requirements(n_structures, avg_atoms_per_structure=50):
       """Estimate HDF5 file size requirements."""
       
       # Base structure data (coordinates, properties)
       base_size_per_structure = avg_atoms_per_structure * 200  # bytes
       
       # Contact data (variable, estimate 100 contacts per structure)
       contact_size_per_structure = 100 * 100  # bytes
       
       # Total estimate
       total_size_mb = (base_size_per_structure + contact_size_per_structure) * n_structures / (1024**2)
       
       # Add 50% overhead for HDF5 metadata and compression efficiency
       estimated_size_mb = total_size_mb * 1.5
       
       logging.info(f"Estimated storage requirement: {estimated_size_mb:.1f} MB")
       return estimated_size_mb

**Parallel Processing Tuning**

.. code-block:: python

   import os
   from concurrent.futures import ProcessPoolExecutor

   def tune_parallel_extraction(n_structures, batch_size):
       """Optimize parallel processing parameters."""
       
       # Calculate number of batches
       n_batches = (n_structures + batch_size - 1) // batch_size
       
       # Optimize worker count
       n_cores = os.cpu_count()
       max_workers = min(
           n_cores - 1,           # Leave one core free
           n_batches,             # No more workers than batches
           8                      # Reasonable upper limit
       )
       
       logging.info(f"Processing {n_structures} structures in {n_batches} batches")
       logging.info(f"Using {max_workers} parallel workers")
       
       return max_workers

See Also
--------

:doc:`../core/crystal_analyzer` : Pipeline orchestration
:doc:`../extraction/structure_post_extraction_processor` : Feature engineering
:doc:`../io/hdf5_utils` : HDF5 utilities
:doc:`../io/data_reader` : Data reading utilities