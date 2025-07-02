dimension_scanner module
=======================

.. automodule:: dimension_scanner
   :members:
   :undoc-members:
   :show-inheritance:

HDF5 Dataset Dimension Analysis and Optimization
-------------------------------------------------

The ``dimension_scanner`` module provides essential functionality for analyzing raw HDF5 structure data to determine optimal array dimensions for efficient batch processing. It scans variable-length datasets to find maximum sizes needed for padding, enabling memory-efficient tensor operations in downstream processing.

**Key Features:**

* **Dimension analysis** - Scan all structures to find maximum array sizes
* **Memory optimization** - Determine minimal padding needed for efficient processing
* **Performance planning** - Estimate memory requirements for batch operations
* **Data validation** - Identify potential issues with dataset sizes
* **Scalability assessment** - Analyze dataset characteristics for processing pipeline design

Core Functions
--------------

.. autofunction:: dimension_scanner.scan_max_dimensions

   **Comprehensive Dataset Dimension Analysis**

   Scans all structures in an HDF5 file to determine the maximum dimensions required for padding ragged arrays into uniform tensors suitable for GPU batch processing.

   **Parameters:**

   * **h5_in** (:obj:`h5py.File`) - Open HDF5 file containing '/structures/<refcode>' groups from raw data extraction
   * **refcodes** (:obj:`List[str]`) - List of structure refcodes to analyze

   **Returns:**

   * **Dict[str, int]** - Dictionary mapping dimension categories to maximum sizes:

     - **'atoms'** (:obj:`int`) - Maximum number of atoms across all structures
     - **'bonds'** (:obj:`int`) - Maximum number of bonds across all structures  
     - **'contacts_inter'** (:obj:`int`) - Maximum intermolecular contacts
     - **'contacts_intra'** (:obj:`int`) - Maximum intramolecular contacts
     - **'hbonds_inter'** (:obj:`int`) - Maximum intermolecular hydrogen bonds
     - **'hbonds_intra'** (:obj:`int`) - Maximum intramolecular hydrogen bonds
     - **'fragments'** (:obj:`int`) - Maximum fragments (set equal to 'atoms')

   **Algorithm Overview:**

   1. **Sequential scanning** - Iterate through all structure groups
   2. **Dataset inspection** - Check sizes of key arrays in each structure
   3. **Maximum tracking** - Track maximum size encountered for each category
   4. **Error handling** - Skip missing datasets with appropriate warnings
   5. **Result compilation** - Return comprehensive dimension dictionary

   **Usage Examples:**

   .. code-block:: python

      import h5py
      from dimension_scanner import scan_max_dimensions

      # Open raw HDF5 file
      with h5py.File('structures_raw.h5', 'r') as h5_file:
          # Get list of all structures
          refcodes = list(h5_file['structures'].keys())
          print(f"Scanning {len(refcodes)} structures...")
          
          # Scan dimensions
          dimensions = scan_max_dimensions(h5_file, refcodes)
          
          # Display results
          print("Maximum dimensions found:")
          for category, max_size in dimensions.items():
              print(f"  {category}: {max_size}")

   **Expected Output Example:**

   .. code-block:: text

      Scanning 1500 structures...
      Maximum dimensions found:
        atoms: 156
        bonds: 168
        contacts_inter: 245
        contacts_intra: 89
        hbonds_inter: 67
        hbonds_intra: 23
        fragments: 156

   **Memory Estimation:**

   .. code-block:: python

      def estimate_memory_requirements(dimensions, batch_size=32, dtype_size=4):
          """Estimate memory requirements based on scanned dimensions."""
          
          # Calculate memory for different data categories (in MB)
          coords_memory = (batch_size * dimensions['atoms'] * 3 * dtype_size) / (1024**2)
          bonds_memory = (batch_size * dimensions['bonds'] * 2 * dtype_size) / (1024**2)
          contacts_memory = (batch_size * dimensions['contacts_inter'] * 6 * dtype_size) / (1024**2)
          
          total_memory = coords_memory + bonds_memory + contacts_memory
          
          print(f"Estimated memory usage for batch_size={batch_size}:")
          print(f"  Atomic coordinates: {coords_memory:.1f} MB")
          print(f"  Bond connectivity: {bonds_memory:.1f} MB") 
          print(f"  Contact data: {contacts_memory:.1f} MB")
          print(f"  Total estimated: {total_memory:.1f} MB")
          
          return total_memory

      # Use scanned dimensions
      total_mem = estimate_memory_requirements(dimensions, batch_size=64)

   **Error Handling:**

   .. code-block:: python

      def robust_dimension_scanning(h5_file, refcodes):
          """Scan dimensions with comprehensive error handling."""
          
          successful_scans = 0
          failed_scans = []
          
          # Track dimensions
          max_dims = {
              'atoms': 0,
              'bonds': 0,
              'contacts_inter': 0,
              'contacts_intra': 0,
              'hbonds_inter': 0,
              'hbonds_intra': 0
          }
          
          for refcode in refcodes:
              try:
                  if refcode not in h5_file['structures']:
                      failed_scans.append((refcode, "Structure group not found"))
                      continue
                  
                  structure_group = h5_file['structures'][refcode]
                  
                  # Check atoms
                  if 'atom_label' in structure_group:
                      n_atoms = structure_group['atom_label'].shape[0]
                      max_dims['atoms'] = max(max_dims['atoms'], n_atoms)
                  
                  # Check bonds (optional)
                  if 'bond_atom1_idx' in structure_group:
                      n_bonds = structure_group['bond_atom1_idx'].shape[0]
                      max_dims['bonds'] = max(max_dims['bonds'], n_bonds)
                  
                  # Check contacts (optional)
                  if 'inter_cc_id' in structure_group:
                      n_contacts = structure_group['inter_cc_id'].shape[0]
                      max_dims['contacts_inter'] = max(max_dims['contacts_inter'], n_contacts)
                  
                  successful_scans += 1
                  
              except Exception as e:
                  failed_scans.append((refcode, str(e)))
          
          # Add fragments dimension
          max_dims['fragments'] = max_dims['atoms']
          
          print(f"Dimension scanning complete:")
          print(f"  Successful: {successful_scans} structures")
          print(f"  Failed: {len(failed_scans)} structures")
          
          if failed_scans:
              print("Failed structures (first 5):")
              for refcode, error in failed_scans[:5]:
                  print(f"  {refcode}: {error}")
          
          return max_dims

