structure_post_extraction_processor module
=========================================

.. automodule:: structure_post_extraction_processor
   :members:
   :undoc-members:
   :show-inheritance:

Advanced Feature Engineering Pipeline
-------------------------------------

The ``structure_post_extraction_processor`` module implements GPU-accelerated computation of advanced structural descriptors and features from raw crystal structure data. This module forms the core of Stage 5 in the CSA pipeline.

Data Structure Classes
----------------------

**CrystalParams**

.. autoclass:: structure_post_extraction_processor.CrystalParams
   :members:
   :undoc-members:
   :show-inheritance:

   **Crystal-Level Parameter Container**

   Dataclass holding crystal properties and unit cell information for batch processing.

   **Attributes:**
      * **identifiers** (:obj:`List[str]`) - CSD refcodes for structures
      * **space_groups** (:obj:`List[str]`) - Space group symbols  
      * **z_values** (:obj:`torch.LongTensor`) - Z values per structure
      * **z_prime** (:obj:`torch.Tensor`) - Z' values per structure
      * **cell_volumes** (:obj:`torch.Tensor`) - Unit cell volumes (Å³)
      * **cell_densities** (:obj:`torch.Tensor`) - Crystal densities (g/cm³)
      * **cell_lengths** (:obj:`torch.Tensor`) - Unit cell parameters a, b, c
      * **cell_angles** (:obj:`torch.Tensor`) - Unit cell angles α, β, γ

**AtomParams**

.. autoclass:: structure_post_extraction_processor.AtomParams
   :members:
   :undoc-members:
   :show-inheritance:

   **Atomic-Level Parameter Container**

   Dataclass holding atomic coordinates, properties, and connectivity information.

   **Attributes:**
      * **labels** (:obj:`List[List[str]]`) - Atomic labels per structure
      * **coords** (:obj:`torch.Tensor`) - Cartesian coordinates (Å)
      * **frac_coords** (:obj:`torch.Tensor`) - Fractional coordinates  
      * **weights** (:obj:`torch.Tensor`) - Atomic masses (Da)
      * **numbers** (:obj:`torch.LongTensor`) - Atomic numbers
      * **charges** (:obj:`torch.Tensor`) - Formal charges
      * **symbols** (:obj:`List[List[str]]`) - Element symbols
      * **sybyl_types** (:obj:`List[List[str]]`) - SYBYL atom types
      * **mask** (:obj:`torch.BoolTensor`) - Validity mask for atoms

**BondParams**

.. autoclass:: structure_post_extraction_processor.BondParams
   :members:
   :undoc-members:
   :show-inheritance:

   **Bond Connectivity Parameter Container**

   Dataclass holding molecular bond information and connectivity graphs.

   **Attributes:**
      * **atom1_labels** (:obj:`List[List[str]]`) - First bond partner labels
      * **atom2_labels** (:obj:`List[List[str]]`) - Second bond partner labels
      * **atom1_idx** (:obj:`torch.LongTensor`) - First partner indices
      * **atom2_idx** (:obj:`torch.LongTensor`) - Second partner indices  
      * **orders** (:obj:`torch.Tensor`) - Bond order values
      * **mask** (:obj:`torch.BoolTensor`) - Validity mask for bonds

**ContactParams**

.. autoclass:: structure_post_extraction_processor.ContactParams
   :members:
   :undoc-members:
   :show-inheritance:

   **Intermolecular Contact Parameter Container**

   Dataclass holding intermolecular close contact information and distances.

   **Attributes:**
      * **central_atom** (:obj:`List[List[str]]`) - Central atom labels
      * **contact_atom** (:obj:`List[List[str]]`) - Contact atom labels
      * **central_atom_idx** (:obj:`torch.LongTensor`) - Central atom indices
      * **contact_atom_idx** (:obj:`torch.LongTensor`) - Contact atom indices
      * **distances** (:obj:`torch.Tensor`) - Contact distances (Å)
      * **mask** (:obj:`torch.BoolTensor`) - Validity mask for contacts

**HBondParams**

.. autoclass:: structure_post_extraction_processor.HBondParams
   :members:
   :undoc-members:
   :show-inheritance:

   **Hydrogen Bond Parameter Container**

   Dataclass holding hydrogen bond geometry and classification information.

   **Attributes:**
      * **central_atom** (:obj:`List[List[str]]`) - Donor atom labels
      * **hydrogen_atom** (:obj:`List[List[str]]`) - Hydrogen atom labels
      * **contact_atom** (:obj:`List[List[str]]`) - Acceptor atom labels
      * **central_atom_idx** (:obj:`torch.LongTensor`) - Donor indices
      * **hydrogen_atom_idx** (:obj:`torch.LongTensor`) - Hydrogen indices
      * **contact_atom_idx** (:obj:`torch.LongTensor`) - Acceptor indices
      * **lengths** (:obj:`torch.Tensor`) - D-A distances (Å)
      * **angles** (:obj:`torch.Tensor`) - D-H-A angles (degrees)
      * **in_los** (:obj:`torch.Tensor`) - Line-of-sight flags
      * **symmetry_A** (:obj:`torch.Tensor`) - Symmetry rotation matrices
      * **symmetry_T** (:obj:`torch.Tensor`) - Symmetry translation vectors

StructurePostExtractionProcessor Class
--------------------------------------

.. autoclass:: structure_post_extraction_processor.StructurePostExtractionProcessor
   :members:
   :undoc-members:
   :show-inheritance:

   **GPU-Accelerated Feature Engineering Pipeline**

   Orchestrates the computation of advanced structural descriptors and features from raw crystal structure data using GPU acceleration for maximum performance.

   **Core Capabilities:**

   * **Rigid Fragment Analysis** - Identifies molecular fragments and rigid groups
   * **Geometric Descriptors** - Computes shape descriptors, moment tensors, orientations
   * **Contact Analysis** - Analyzes intermolecular interactions and networks  
   * **Symmetry Operations** - Applies crystallographic symmetry transformations
   * **Feature Engineering** - Derives machine learning-ready descriptors
   * **GPU Acceleration** - Leverages CUDA for batch tensor operations

   **Processing Pipeline:**

   1. **Data Loading** - Read raw data from HDF5 into GPU tensors
   2. **Fragment Identification** - Detect rigid molecular fragments
   3. **Geometric Analysis** - Compute centers of mass, inertia tensors, orientations
   4. **Contact Expansion** - Apply symmetry operations to intermolecular contacts
   5. **Feature Computation** - Calculate advanced descriptors and properties
   6. **Data Writing** - Save computed features to processed HDF5 file

   **Attributes:**
      * **hdf5_path** (:obj:`Path`) - Input HDF5 file with raw data
      * **batch_size** (:obj:`int`) - Structures processed per GPU batch
      * **device** (:obj:`torch.device`) - GPU device for tensor operations
      * **raw_reader** (:obj:`DataReader`) - Raw data loading interface
      * **raw_writer** (:obj:`DataWriter`) - Raw data writing interface
      * **computed_writer** (:obj:`DataWriter`) - Computed data writing interface

   .. automethod:: __init__

      **Initialize Post-Extraction Processor**

      Parameters:
         * **hdf5_path** (:obj:`Path`) - Path to raw HDF5 file
         * **batch_size** (:obj:`int`) - Number of structures per GPU batch
         * **device** (:obj:`Optional[Union[str, torch.device]]`) - GPU device specification

      **Device Configuration:**

      .. code-block:: python

         # Automatic device selection
         if device is None:
             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         
         # Manual device specification
         device = torch.device("cuda:0")  # Specific GPU
         device = torch.device("cpu")     # Force CPU processing

      **Memory Management:**

      .. code-block:: python

         # Configure GPU memory settings
         if device.type == "cuda":
             torch.cuda.empty_cache()
             torch.cuda.set_per_process_memory_fraction(0.8)

      **File Initialization:**

      .. code-block:: python

         # Set up data readers and writers
         self.raw_reader = DataReader(hdf5_path)
         
         # Create processed data file
         processed_path = hdf5_path.with_suffix("_processed.h5")
         self.raw_writer = DataWriter(processed_path, "raw")
         self.computed_writer = DataWriter(processed_path, "computed")

   .. automethod:: run

      **Execute Complete Feature Engineering Pipeline**

      Orchestrates the full post-extraction processing workflow from raw data loading to computed feature storage.

      **Processing Workflow:**

      1. **Initialization** - Set up output files and data structures
      2. **Batch Processing** - Process structures in GPU-optimized batches
      3. **Feature Computation** - Apply all feature engineering algorithms
      4. **Data Writing** - Store results in structured HDF5 format
      5. **Validation** - Verify data integrity and completeness

      **Batch Processing Loop:**

      .. code-block:: python

         for batch_start in range(0, total_structures, self.batch_size):
             batch_end = min(batch_start + self.batch_size, total_structures)
             
             # Load raw data batch
             raw_data = self.raw_reader.load_batch(batch_start, batch_end)
             
             # Process on GPU
             with torch.cuda.device(self.device):
                 computed_data = self._process_batch(raw_data)
             
             # Write results
             self._write_batch_results(batch_start, raw_data, computed_data)
             
             # Memory cleanup
             torch.cuda.empty_cache()

      **Progress Monitoring:**

      .. code-block:: text

         INFO - Starting post-extraction processing...
         INFO - Found 8234 structures to process
         INFO - Processing batch 1/258 (32 structures)
         INFO - Computed 1024 rigid fragments
         INFO - Processed 15678 intermolecular contacts
         INFO - Processing batch 2/258 (32 structures)
         ...
         INFO - Post-extraction processing complete
         INFO - Processed file saved: analysis_processed.h5

      **GPU Memory Management:**

      .. code-block:: python

         def monitor_gpu_memory():
             if torch.cuda.is_available():
                 allocated = torch.cuda.memory_allocated() / 1024**3
                 cached = torch.cuda.memory_reserved() / 1024**3
                 logging.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

      **Error Handling:**

      * **GPU OOM** - Automatically reduces batch size and retries
      * **Data Corruption** - Skips problematic structures with logging
      * **Device Failures** - Falls back to CPU processing if needed
      * **File I/O Issues** - Implements robust error recovery

   .. automethod:: _process_batch

      **Process Single Batch with GPU Acceleration**

      Core batch processing function that applies all feature engineering algorithms to a batch of structures.

      **Parameters:**
         * **start** (:obj:`int`) - Starting structure index in batch
         * **crystal** (:obj:`CrystalParams`) - Crystal properties
         * **atom** (:obj:`AtomParams`) - Atomic data
         * **bond** (:obj:`BondParams`) - Bond connectivity
         * **intra_cc** (:obj:`ContactParams`) - Intramolecular contacts
         * **intra_hb** (:obj:`HBondParams`) - Intramolecular H-bonds
         * **inter_cc** (:obj:`ContactParams`) - Intermolecular contacts
         * **inter_hb** (:obj:`HBondParams`) - Intermolecular H-bonds

      **Feature Engineering Pipeline:**

      .. code-block:: python

         # 1. Compute unit cell transformation matrices
         cell_matrices = self._compute_cell_matrices(
             crystal.cell_lengths, crystal.cell_angles
         )
         
         # 2. Identify rotatable bonds
         rotatable_bonds = self._compute_rotatable_bonds(
             atom.symbols, bond.atom1_idx, bond.atom2_idx, bond.orders
         )
         
         # 3. Expand intermolecular contacts with symmetry
         expanded_contacts = self._expand_inter_contacts(
             inter_cc, cell_matrices
         )
         
         # 4. Identify rigid molecular fragments
         fragment_ids = self._compute_rigid_fragments(
             atom.mask, bond.atom1_idx, bond.atom2_idx, rotatable_bonds
         )
         
         # 5. Compute fragment properties
         fragment_properties = self._compute_fragment_properties(
             fragment_ids, atom.coords, atom.weights, atom.charges
         )
         
         # 6. Map contacts to fragments
         contact_fragments = self._identify_contact_fragments(
             expanded_contacts, fragment_ids
         )

Fragment Analysis Methods
-------------------------

.. automethod:: structure_post_extraction_processor.StructurePostExtractionProcessor._compute_rigid_fragments

   **Identify Rigid Molecular Fragments**

   Analyzes molecular connectivity to identify rigid groups of atoms that move as single units.

   **Parameters:**
      * **atom_mask** (:obj:`torch.BoolTensor`) - Valid atom flags
      * **bond_atom1** (:obj:`torch.LongTensor`) - First bond partner indices
      * **bond_atom2** (:obj:`torch.LongTensor`) - Second bond partner indices  
      * **rotatable_mask** (:obj:`torch.BoolTensor`) - Rotatable bond flags

   **Algorithm:**

   1. **Graph Construction** - Build molecular connectivity graph
   2. **Bond Classification** - Identify rotatable vs. rigid bonds
   3. **Component Analysis** - Find connected components after removing rotatable bonds
   4. **Fragment Assignment** - Assign unique IDs to each rigid fragment

   **Fragment Types Identified:**

   * **Aromatic Rings** - Benzene, pyridine, etc.
   * **Aliphatic Rings** - Cyclohexane, cyclopentane, etc.
   * **Rigid Chains** - Double/triple bonded segments
   * **Individual Atoms** - Isolated atoms or small groups

   **Returns:**
      :obj:`torch.LongTensor` with fragment IDs for each atom

.. automethod:: structure_post_extraction_processor.StructurePostExtractionProcessor._compute_fragment_properties

   **Compute Geometric Properties of Fragments**

   Calculates comprehensive geometric descriptors for each identified rigid fragment.

   **Properties Computed:**

   **Centers and Centroids:**
   * **Center of Mass** - Mass-weighted average position
   * **Geometric Centroid** - Unweighted average position
   * **Charge Centroid** - Charge-weighted average position

   **Inertia Analysis:**
   * **Inertia Tensor** - 3×3 moment of inertia matrix
   * **Principal Moments** - Eigenvalues of inertia tensor
   * **Principal Axes** - Eigenvectors defining orientation
   * **Gyration Radius** - Root-mean-square distance from COM

   **Shape Descriptors:**
   * **Asphericity** - Deviation from spherical shape
   * **Acylindricity** - Deviation from cylindrical shape  
   * **Shape Anisotropy** - Overall shape asymmetry
   * **Relative Shape** - Normalized shape parameters

   **Geometric Extents:**
   * **Bounding Box** - Axis-aligned minimum bounding box
   * **Principal Extents** - Extents along principal axes
   * **Surface Area** - Molecular surface area estimate
   * **Volume** - van der Waals volume estimate

Geometric Computation Methods
-----------------------------

.. automethod:: structure_post_extraction_processor.StructurePostExtractionProcessor._compute_fragment_com_centroid

   **Compute Fragment Centers of Mass and Centroids**

   Calculates both mass-weighted and geometric centers for fragments.

   **Parameters:**
      * **fragment_coords** (:obj:`torch.Tensor`) - Atomic coordinates per fragment
      * **fragment_frac_coords** (:obj:`torch.Tensor`) - Fractional coordinates
      * **fragment_weights** (:obj:`torch.Tensor`) - Atomic masses
      * **fragment_mask** (:obj:`torch.BoolTensor`) - Valid atom flags

   **Center of Mass Calculation:**

   .. code-block:: python

      # Mass-weighted center calculation
      total_mass = torch.sum(fragment_weights * fragment_mask, dim=-1)
      weighted_coords = fragment_coords * fragment_weights.unsqueeze(-1)
      com_coords = torch.sum(weighted_coords * fragment_mask.unsqueeze(-1), dim=-2) / total_mass.unsqueeze(-1)

   **Geometric Centroid:**

   .. code-block:: python

      # Unweighted geometric center
      n_atoms = torch.sum(fragment_mask, dim=-1)
      centroid_coords = torch.sum(fragment_coords * fragment_mask.unsqueeze(-1), dim=-2) / n_atoms.unsqueeze(-1)

   **Returns:**
      Dictionary with computed centers in both Cartesian and fractional coordinates

.. automethod:: structure_post_extraction_processor.StructurePostExtractionProcessor._compute_fragment_inertia_tensors

   **Compute Inertia Tensors and Principal Moments**

   Calculates inertia tensors, eigenvalues, and eigenvectors for fragment orientation analysis.

   **Inertia Tensor Calculation:**

   .. code-block:: python

      # Translate to center of mass
      coords_centered = fragment_coords - fragment_com.unsqueeze(-2)
      
      # Compute inertia tensor components
      I_xx = torch.sum(weights * (coords_centered[..., 1]**2 + coords_centered[..., 2]**2), dim=-1)
      I_yy = torch.sum(weights * (coords_centered[..., 0]**2 + coords_centered[..., 2]**2), dim=-1)
      I_zz = torch.sum(weights * (coords_centered[..., 0]**2 + coords_centered[..., 1]**2), dim=-1)
      I_xy = -torch.sum(weights * coords_centered[..., 0] * coords_centered[..., 1], dim=-1)
      I_xz = -torch.sum(weights * coords_centered[..., 0] * coords_centered[..., 2], dim=-1)
      I_yz = -torch.sum(weights * coords_centered[..., 1] * coords_centered[..., 2], dim=-1)
      
      # Assemble 3x3 inertia tensor
      inertia_tensor = torch.stack([
          torch.stack([I_xx, I_xy, I_xz], dim=-1),
          torch.stack([I_xy, I_yy, I_yz], dim=-1),
          torch.stack([I_xz, I_yz, I_zz], dim=-1)
      ], dim=-2)

   **Principal Moment Analysis:**

   .. code-block:: python

      # Eigendecomposition for principal moments
      eigenvalues, eigenvectors = torch.linalg.eigh(inertia_tensor)
      
      # Sort by magnitude (ascending)
      sorted_indices = torch.argsort(eigenvalues, dim=-1)
      principal_moments = torch.gather(eigenvalues, -1, sorted_indices)
      principal_axes = torch.gather(eigenvectors, -1, sorted_indices.unsqueeze(-2).expand(-1, -1, 3, -1))

   **Shape Parameters:**

   .. code-block:: python

      # Asphericity: measure of deviation from sphere
      I1, I2, I3 = principal_moments[..., 0], principal_moments[..., 1], principal_moments[..., 2]
      asphericity = I3 - 0.5 * (I1 + I2)
      
      # Acylindricity: measure of deviation from cylinder  
      acylindricity = I2 - I1
      
      # Relative shape anisotropy
      shape_anisotropy = (asphericity**2 + 0.75 * acylindricity**2) / (I1 + I2 + I3)**2

   **Returns:**
      Dictionary with inertia tensors, principal moments, axes, and shape descriptors

Contact Analysis Methods
------------------------

.. automethod:: structure_post_extraction_processor.StructurePostExtractionProcessor._expand_inter_contacts

   **Expand Intermolecular Contacts with Symmetry Operations**

   Applies crystallographic symmetry operations to generate the complete intermolecular contact network.

   **Parameters:**
      * **contacts** (:obj:`ContactParams`) - Raw intermolecular contacts
      * **cell_matrices** (:obj:`torch.Tensor`) - Unit cell transformation matrices

   **Symmetry Expansion Process:**

   .. code-block:: python

      # Parse symmetry operators from contact data
      symmetry_ops = parse_symmetry_operators(contact_symmetry_strings)
      
      # Apply rotation matrices
      rotation_matrices = symmetry_ops['rotation']  # Shape: (B, C, 3, 3)
      translation_vectors = symmetry_ops['translation']  # Shape: (B, C, 3)
      
      # Transform contact atom coordinates
      contact_coords_transformed = torch.matmul(
          rotation_matrices, 
          contact_coords.unsqueeze(-1)
      ).squeeze(-1) + translation_vectors
      
      # Convert to Cartesian coordinates
      contact_coords_cartesian = torch.matmul(
          contact_coords_transformed, 
          cell_matrices
      )

   **Distance Recalculation:**

   .. code-block:: python

      # Compute actual intermolecular distances
      distance_vectors = contact_coords_cartesian - central_coords
      distances = torch.norm(distance_vectors, dim=-1)
      
      # Verify contact validity (within cutoff distance)
      valid_contacts = distances < contact_cutoff_distance

   **Contact Classification:**

   * **van der Waals Contacts** - Within vdW radii sum + tolerance
   * **Close Contacts** - Shorter than vdW sum (potential strain)
   * **Hydrogen Bonds** - Specific geometric criteria
   * **π-π Interactions** - Aromatic ring stacking
   * **Electrostatic Contacts** - Charge-charge interactions

   **Returns:**
      Dictionary with expanded contact coordinates, distances, and classifications

.. automethod:: structure_post_extraction_processor.StructurePostExtractionProcessor._flag_hbond_contacts

   **Identify Hydrogen Bond Contacts**

   Determines which close contacts are actually part of hydrogen bonds based on geometric criteria.

   **Parameters:**
      * **cc_central_idx** (:obj:`torch.LongTensor`) - Contact central atom indices
      * **cc_contact_idx** (:obj:`torch.LongTensor`) - Contact contact atom indices
      * **cc_mask** (:obj:`torch.BoolTensor`) - Contact validity mask
      * **hb_central_idx** (:obj:`torch.LongTensor`) - H-bond donor indices
      * **hb_hydrogen_idx** (:obj:`torch.LongTensor`) - H-bond hydrogen indices
      * **hb_contact_idx** (:obj:`torch.LongTensor`) - H-bond acceptor indices
      * **hb_mask** (:obj:`torch.BoolTensor`) - H-bond validity mask

   **Classification Algorithm:**

   .. code-block:: python

      # Match contacts to hydrogen bonds
      hbond_flags = torch.zeros_like(cc_mask, dtype=torch.bool)
      
      for b in range(batch_size):
          for c in range(max_contacts):
              if not cc_mask[b, c]:
                  continue
                  
              central_atom = cc_central_idx[b, c]
              contact_atom = cc_contact_idx[b, c]
              
              # Check if this contact matches any H-bond
              for h in range(max_hbonds):
                  if not hb_mask[b, h]:
                      continue
                      
                  # Match donor-acceptor pair
                  if (hb_central_idx[b, h] == central_atom and 
                      hb_contact_idx[b, h] == contact_atom):
                      hbond_flags[b, c] = True
                      break

   **Geometric Criteria:**

   * **Distance Cutoff** - D-A distance < 3.5 Å
   * **Angular Cutoff** - D-H-A angle > 120°
   * **Linearity** - Preference for linear arrangements
   * **Chemical Validation** - Appropriate donor/acceptor atoms

   **Returns:**
      :obj:`torch.BoolTensor` indicating which contacts are H-bonds

.. automethod:: structure_post_extraction_processor.StructurePostExtractionProcessor._identify_contact_fragments

   **Map Contacts to Fragment Pairs**

   Associates each intermolecular contact with the rigid fragments involved.

   **Parameters:**
      * **cc_central_idx** (:obj:`torch.LongTensor`) - Central atom indices
      * **cc_contact_idx** (:obj:`torch.LongTensor`) - Contact atom indices  
      * **atom_fragment_id** (:obj:`torch.LongTensor`) - Fragment ID per atom

   **Fragment Mapping:**

   .. code-block:: python

      # Map contact atoms to their fragments
      central_fragment_ids = torch.gather(
          atom_fragment_id, 
          dim=-1, 
          index=cc_central_idx
      )
      
      contact_fragment_ids = torch.gather(
          atom_fragment_id,
          dim=-1, 
          index=cc_contact_idx
      )
      
      # Create fragment pair identifiers
      fragment_pairs = torch.stack([
          central_fragment_ids, 
          contact_fragment_ids
      ], dim=-1)

   **Contact Statistics:**

   * **Contacts per Fragment** - Number of intermolecular contacts
   * **Fragment Coordination** - Number of neighboring fragments
   * **Contact Directionality** - Preferred contact directions
   * **Fragment Accessibility** - Solvent-accessible surface contacts

   **Returns:**
      Dictionary with fragment indices and contact-fragment mappings

Advanced Feature Computation
-----------------------------

.. automethod:: structure_post_extraction_processor.StructurePostExtractionProcessor._compute_contact_com_vectors

   **Compute Contact-to-COM Vectors and Distances**

   Calculates vectors and distances from contact atoms to fragment centers of mass.

   **Parameters:**
      * **cc_coords** (:obj:`torch.Tensor`) - Contact atom coordinates
      * **cc_frac_coords** (:obj:`torch.Tensor`) - Contact fractional coordinates
      * **cc_fragment_idx** (:obj:`torch.LongTensor`) - Contact fragment indices
      * **frag_com_coords** (:obj:`torch.Tensor`) - Fragment COM coordinates
      * **frag_com_frac_coords** (:obj:`torch.Tensor`) - Fragment COM fractional coordinates
      * **frag_structure_id** (:obj:`torch.LongTensor`) - Fragment structure IDs
      * **frag_local_ids** (:obj:`torch.LongTensor`) - Fragment local IDs

   **Vector Calculations:**

   .. code-block:: python

      # Map contacts to fragment COM positions
      contact_com_coords = torch.gather(
          frag_com_coords,
          dim=-2,
          index=cc_fragment_idx.unsqueeze(-1).expand(-1, -1, 3)
      )
      
      # Compute contact-to-COM vectors
      com_vectors = cc_coords - contact_com_coords
      com_distances = torch.norm(com_vectors, dim=-1)
      
      # Normalize for directional analysis
      com_unit_vectors = com_vectors / com_distances.unsqueeze(-1)

   **Geometric Descriptors:**

   * **Contact Distance** - Direct atom-atom distance
   * **COM Distance** - Distance from contact to fragment center
   * **Contact Vector** - Direction from COM to contact point
   * **Surface Normal** - Estimated surface normal at contact
   * **Contact Accessibility** - Geometric accessibility measure

   **Applications:**

   * **Packing Analysis** - How fragments pack together
   * **Surface Properties** - Contact surface characterization
   * **Intermolecular Forces** - Direction and magnitude analysis
   * **Crystal Engineering** - Design principles extraction

   **Returns:**
      Dictionary with contact vectors, distances, and geometric descriptors

Data Writing and I/O Methods
-----------------------------

.. automethod:: structure_post_extraction_processor.StructurePostExtractionProcessor._write_batch_results

   **Write Computed Features to HDF5**

   Stores both raw and computed data for a processed batch in structured HDF5 format.

   **Data Organization:**

   .. code-block:: text

      analysis_processed.h5
      ├── /raw/                          # Copy of original raw data
      │   ├── crystal_data/
      │   ├── atom_data/
      │   ├── bond_data/
      │   └── contact_data/
      └── /computed/                     # Computed features
          ├── crystal_data/
          │   ├── cell_matrices          # Unit cell transformation matrices
          │   ├── reciprocal_matrices    # Reciprocal space matrices
          │   └── symmetry_matrices      # Space group operators
          ├── atom_data/
          │   ├── fragment_ids           # Rigid fragment assignments
          │   ├── heavy_atom_mask        # Non-hydrogen atom flags
          │   └── coordination_numbers   # Atomic coordination
          ├── fragment_data/
          │   ├── com_coords             # Centers of mass
          │   ├── centroid_coords        # Geometric centroids
          │   ├── inertia_tensors        # Moment of inertia tensors
          │   ├── principal_moments      # Principal moment eigenvalues
          │   ├── principal_axes         # Principal axis eigenvectors
          │   ├── gyration_radii         # Radii of gyration
          │   ├── asphericity            # Shape asphericity
          │   ├── acylindricity          # Shape acylindricity
          │   └── shape_anisotropy       # Overall shape anisotropy
          ├── contact_data/
          │   ├── expanded_coords        # Symmetry-expanded coordinates
          │   ├── contact_distances      # Actual contact distances
          │   ├── contact_vectors        # Contact direction vectors
          │   ├── is_hbond              # Hydrogen bond flags
          │   ├── fragment_indices       # Contact fragment mappings
          │   └── com_distances          # Distance to fragment COM
          └── hbond_data/
              ├── expanded_geometry      # Symmetry-expanded H-bond geometry
              ├── contact_mapping        # H-bond to contact mapping
              └── geometric_parameters   # Additional geometric descriptors

   **Writing Process:**

   .. code-block:: python

      # Write raw data (copy from input)
      self.raw_writer.write_raw_crystal_data(start, crystal_parameters)
      self.raw_writer.write_raw_atom_data(start, atom_parameters)
      self.raw_writer.write_raw_bond_data(start, bond_parameters)
      
      # Write computed features
      self.computed_writer.write_computed_crystal_data(start, computed_crystal)
      self.computed_writer.write_computed_fragment_data(start, computed_fragments)
      self.computed_writer.write_computed_contact_data(start, computed_contacts)

   **Data Type Optimization:**

   .. code-block:: python

      # Use appropriate precision for different data types
      coordinates = computed_data['coordinates'].cpu().numpy().astype(np.float32)
      distances = computed_data['distances'].cpu().numpy().astype(np.float32)
      indices = computed_data['indices'].cpu().numpy().astype(np.int32)
      flags = computed_data['flags'].cpu().numpy().astype(bool)

Usage Examples
--------------

**Basic Feature Engineering**

.. code-block:: python

   from structure_post_extraction_processor import StructurePostExtractionProcessor
   import torch
   from pathlib import Path

   # Initialize processor
   processor = StructurePostExtractionProcessor(
       hdf5_path=Path("./analysis.h5"),
       batch_size=32,
       device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
   )

   # Execute feature engineering
   processor.run()

**GPU Memory Optimization**

.. code-block:: python

   # Configure for large datasets with limited GPU memory
   import torch

   # Set memory fraction
   torch.cuda.set_per_process_memory_fraction(0.7)
   
   # Use smaller batch size
   memory_optimized_processor = StructurePostExtractionProcessor(
       hdf5_path=Path("./large_dataset.h5"),
       batch_size=16,  # Reduced batch size
       device=torch.device("cuda")
   )
   
   # Monitor memory usage
   def memory_monitor():
       allocated = torch.cuda.memory_allocated() / 1024**3
       cached = torch.cuda.memory_reserved() / 1024**3
       print(f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
   
   # Run with monitoring
   memory_monitor()
   memory_optimized_processor.run()
   memory_monitor()

**Custom Feature Selection**

.. code-block:: python

   # Subclass for custom feature computation
   class CustomFeatureProcessor(StructurePostExtractionProcessor):
       
       def _process_batch(self, start, crystal, atom, bond, intra_cc, intra_hb, inter_cc, inter_hb):
           """Override to add custom features."""
           
           # Call parent processing
           super()._process_batch(start, crystal, atom, bond, intra_cc, intra_hb, inter_cc, inter_hb)
           
           # Add custom features
           custom_features = self._compute_custom_descriptors(
               atom.coords, atom.symbols, bond.atom1_idx, bond.atom2_idx
           )
           
           # Write custom features
           self._write_custom_features(start, custom_features)
       
       def _compute_custom_descriptors(self, coords, symbols, bond1, bond2):
           """Compute application-specific descriptors."""
           
           # Example: Compute molecular surface area
           surface_areas = self._compute_molecular_surface_area(coords, symbols)
           
           # Example: Compute ring strain energies
           ring_strains = self._compute_ring_strain_energies(coords, bond1, bond2)
           
           return {
               'surface_areas': surface_areas,
               'ring_strains': ring_strains
           }

**High-Throughput Processing**

.. code-block:: python

   # Configure for maximum throughput
   high_throughput_processor = StructurePostExtractionProcessor(
       hdf5_path=Path("./massive_dataset.h5"),
       batch_size=128,  # Large batches for throughput
       device=torch.device("cuda")
   )
   
   # Use multiple GPUs if available
   if torch.cuda.device_count() > 1:
       # Implement data parallel processing
       import torch.nn as nn
       
       class ParallelProcessor(nn.DataParallel):
           def __init__(self, processor):
               super().__init__(processor)
           
           def forward(self, batch_data):
               return self.module._process_batch_parallel(batch_data)
       
       parallel_processor = ParallelProcessor(high_throughput_processor)

**Quality Control and Validation**

.. code-block:: python

   # Implement comprehensive validation
   class ValidatedProcessor(StructurePostExtractionProcessor):
       
       def _validate_batch_results(self, computed_data):
           """Validate computed features for quality."""
           
           # Check for NaN values
           for key, tensor in computed_data.items():
               if torch.isnan(tensor).any():
                   logging.warning(f"NaN values detected in {key}")
           
           # Check physical constraints
           com_distances = computed_data.get('com_distances')
           if com_distances is not None:
               if (com_distances < 0).any():
                   logging.error("Negative distances detected")
           
           # Check fragment validity
           fragment_ids = computed_data.get('fragment_ids')
           if fragment_ids is not None:
               max_fragment_id = fragment_ids.max()
               expected_max = len(torch.unique(fragment_ids)) - 1
               if max_fragment_id != expected_max:
                   logging.warning("Fragment ID inconsistency detected")
       
       def _process_batch(self, *args, **kwargs):
           # Process normally
           result = super()._process_batch(*args, **kwargs)
           
           # Validate results
           self._validate_batch_results(result)
           
           return result

Performance Optimization
------------------------

**GPU Utilization**

.. code-block:: python

   # Optimize GPU kernel launches
   torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
   torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
   
   # Use mixed precision for memory efficiency
   from torch.cuda.amp import autocast, GradScaler
   
   class MixedPrecisionProcessor(StructurePostExtractionProcessor):
       
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.scaler = GradScaler()
       
       def _process_batch(self, *args, **kwargs):
           with autocast():
               return super()._process_batch(*args, **kwargs)

**Memory Management**

.. code-block:: python

   # Implement gradient checkpointing for memory efficiency
   import torch.utils.checkpoint as checkpoint
   
   def memory_efficient_feature_computation(coords, weights):
       """Use checkpointing for memory-intensive computations."""
       
       def compute_inertia_checkpoint(coords_chunk, weights_chunk):
           return compute_inertia_tensor(coords_chunk, weights_chunk)
       
       # Use checkpointing to trade compute for memory
       inertia_tensors = checkpoint.checkpoint(
           compute_inertia_checkpoint, 
           coords, 
           weights,
           use_reentrant=False
       )
       
       return inertia_tensors

**Batch Size Optimization**

.. code-block:: python

   def find_optimal_batch_size(processor, max_batch_size=256):
       """Find maximum feasible batch size through binary search."""
       
       low, high = 1, max_batch_size
       optimal_size = 1
       
       while low <= high:
           mid = (low + high) // 2
           
           try:
               # Test with current batch size
               processor.batch_size = mid
               test_batch = processor._create_test_batch()
               processor._process_batch(*test_batch)
               
               # Success - try larger
               optimal_size = mid
               low = mid + 1
               
           except RuntimeError as e:
               if "out of memory" in str(e):
                   # OOM - try smaller
                   high = mid - 1
                   torch.cuda.empty_cache()
               else:
                   raise
       
       return optimal_size

See Also
--------

:doc:`../core/crystal_analyzer` : Pipeline orchestration
:doc:`../extraction/structure_data_extractor` : Raw data extraction
:doc:`../processing/fragment_utils` : Fragment analysis utilities
:doc:`../processing/geometry_utils` : Geometric computation utilities
:doc:`../io/data_writer` : HDF5 data writing utilities