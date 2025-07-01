geometry_utils module
=====================

.. automodule:: geometry_utils
   :members:
   :undoc-members:
   :show-inheritance:

Geometric Calculations and Descriptors
---------------------------------------

The ``geometry_utils`` module provides GPU-accelerated batch computations for molecular and crystallographic geometric descriptors. All functions operate on PyTorch tensors and support batch processing for high-throughput analysis.

**Key Features:**

* **Bond geometry analysis** - angles, torsions, planarity metrics
* **Crystallographic calculations** - distances to special planes, order parameters  
* **Molecular descriptors** - inertia tensors, quaternions, shape parameters
* **GPU acceleration** - optimized PyTorch operations for large datasets
* **Batch processing** - efficient handling of multiple structures simultaneously

Core Functions
--------------

Bond Angle Calculations
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: geometry_utils.compute_bond_angles_batch

   **Comprehensive Bond Angle Analysis**

   Computes all unique bond angles (i–j–k) where atom j is the central vertex connected to both i and k.

   **Algorithm:**
   
   1. Build adjacency graph from bond connectivity
   2. Identify all valid angle triplets (i–j–k)
   3. Compute vectorized angle calculations using dot products
   4. Return organized results with proper masking

   **Parameters:**

   * **atom_labels** (:obj:`List[List[str]]`) - Atom labels per structure for identification
   * **atom_coords** (:obj:`torch.Tensor`, shape (B, N, 3)) - Cartesian coordinates  
   * **atom_mask** (:obj:`torch.BoolTensor`, shape (B, N)) - Valid atom indicators
   * **bond_atom1_idx** (:obj:`torch.LongTensor`, shape (B, M)) - First bond atom indices
   * **bond_atom2_idx** (:obj:`torch.LongTensor`, shape (B, M)) - Second bond atom indices
   * **bond_mask** (:obj:`torch.BoolTensor`, shape (B, M)) - Valid bond indicators
   * **device** (:obj:`torch.device`) - GPU/CPU device for computation

   **Returns:**

   * **angle_ids** (:obj:`List[List[str]]`) - Angle identifiers as "i–j–k" strings
   * **angles** (:obj:`torch.Tensor`, shape (B, P_max)) - Angle values in degrees
   * **mask_ang** (:obj:`torch.BoolTensor`, shape (B, P_max)) - Valid angle indicators  
   * **idx_tensor** (:obj:`torch.LongTensor`, shape (B, P_max, 3)) - Atom index triplets

   **Usage Example:**

   .. code-block:: python

      import torch
      from geometry_utils import compute_bond_angles_batch

      # Sample molecular data
      atom_labels = [['C1', 'C2', 'C3', 'H1']]
      coords = torch.tensor([[[0.0, 0.0, 0.0],    # C1
                              [1.4, 0.0, 0.0],    # C2  
                              [2.1, 1.2, 0.0],    # C3
                              [3.2, 1.2, 0.0]]]).float()  # H1
      atom_mask = torch.tensor([[True, True, True, True]])
      
      # Bond connectivity: C1-C2, C2-C3, C3-H1
      bond_atom1 = torch.tensor([[0, 1, 2]])  
      bond_atom2 = torch.tensor([[1, 2, 3]])
      bond_mask = torch.tensor([[True, True, True]])
      
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      
      angle_ids, angles, mask_ang, idx_tensor = compute_bond_angles_batch(
          atom_labels, coords, atom_mask, 
          bond_atom1, bond_atom2, bond_mask, device
      )
      
      print(f"Found {mask_ang.sum().item()} valid angles:")
      for i, angle_id in enumerate(angle_ids[0]):
          if mask_ang[0, i]:
              print(f"  {angle_id}: {angles[0, i]:.1f}°")

   **Performance Notes:**

   * Scales as O(B × M²) where M is the maximum number of bonds per structure
   * GPU acceleration provides 10-50× speedup over CPU for large batches
   * Memory usage: ~4 bytes per angle × batch_size × max_angles

Torsion Angle Calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: geometry_utils.compute_torsion_angles_batch

   **Dihedral Angle Computation**

   Calculates all valid torsion (dihedral) angles for molecular conformations using the four-atom sequence i–j–k–l.

   **Parameters:**

   * **atom_labels** (:obj:`List[List[str]]`) - Atom identification labels
   * **atom_coords** (:obj:`torch.Tensor`, shape (B, N, 3)) - 3D atomic coordinates
   * **atom_mask** (:obj:`torch.BoolTensor`, shape (B, N)) - Valid atom mask
   * **bond_atom1_idx** (:obj:`torch.LongTensor`, shape (B, M)) - Bond connectivity indices
   * **bond_atom2_idx** (:obj:`torch.LongTensor`, shape (B, M)) - Bond connectivity indices  
   * **bond_mask** (:obj:`torch.BoolTensor`, shape (B, M)) - Valid bond mask
   * **device** (:obj:`torch.device`) - Computation device

   **Returns:**

   * **torsion_ids** (:obj:`List[List[str]]`) - Torsion identifiers as "i–j–k–l"
   * **torsions** (:obj:`torch.Tensor`, shape (B, T_max)) - Dihedral angles in degrees (-180° to +180°)
   * **mask_tor** (:obj:`torch.BoolTensor`, shape (B, T_max)) - Valid torsion mask
   * **idx_tensor** (:obj:`torch.LongTensor`, shape (B, T_max, 4)) - Four-atom index sets

   **Mathematical Implementation:**

   Uses the standard dihedral angle formula with cross products:

   .. math::

      \phi = \arctan2\left(\vec{n_1} \times \vec{n_2} \cdot \hat{b_2}, \vec{n_1} \cdot \vec{n_2}\right)

   Where :math:`\vec{n_1} = \vec{b_1} \times \vec{b_2}` and :math:`\vec{n_2} = \vec{b_2} \times \vec{b_3}`.

Bond Rotatability Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: geometry_utils.compute_bond_rotatability_batch

   **Rotatable Bond Classification**

   Identifies rotatable bonds based on chemical environment and structural constraints.

   **Rotatable Bond Criteria:**

   1. **Non-hydrogen connectivity** - Both atoms have ≥2 non-hydrogen neighbors
   2. **Single bond type** - Must be a single covalent bond
   3. **Non-cyclic** - Bond is not part of a ring system
   4. **Non-linear** - Neither atom is sp-hybridized or in cumulated double bonds

   **Parameters:**

   * **atom_symbols** (:obj:`List[List[str]]`) - Atomic symbols (C, N, O, etc.)
   * **bond_atom1_idx** (:obj:`torch.LongTensor`) - Bond connectivity
   * **bond_atom2_idx** (:obj:`torch.LongTensor`) - Bond connectivity
   * **bond_types** (:obj:`List[List[str]]`) - Bond type annotations ('single', 'double', etc.)
   * **bond_in_ring** (:obj:`List[List[bool]]`) - Ring membership flags
   * **bond_mask** (:obj:`torch.BoolTensor`) - Valid bond indicators
   * **device** (:obj:`torch.device`) - Computation device

   **Returns:**

   * **torch.BoolTensor**, shape (B, M) - True for rotatable bonds

   **Applications:**

   * Drug-like property assessment (Lipinski's Rule of Five)
   * Conformational flexibility analysis  
   * Molecular dynamics preparation
   * Structure-activity relationship studies

Planarity Analysis
~~~~~~~~~~~~~~~~~~

.. autofunction:: geometry_utils.compute_best_fit_plane_batch

   **Best-Fit Plane Computation**

   Determines optimal plane through a set of atoms using least-squares fitting with optional weighting.

   **Parameters:**

   * **coords** (:obj:`torch.Tensor`, shape (B, N, 3)) - Atomic coordinates
   * **weights** (:obj:`torch.Tensor`, shape (B, N)) - Per-atom weights (masses, charges, etc.)
   * **mask** (:obj:`torch.BoolTensor`, shape (B, N)) - Valid atom indicators
   * **device** (:obj:`torch.device`) - Computation device

   **Returns:**

   * **plane_normal** (:obj:`torch.Tensor`, shape (B, 3)) - Unit normal vectors
   * **plane_centroid** (:obj:`torch.Tensor`, shape (B, 3)) - Plane centroids
   * **eigenvalues** (:obj:`torch.Tensor`, shape (B, 3)) - Principal component eigenvalues

.. autofunction:: geometry_utils.compute_planarity_metrics_batch

   **Comprehensive Planarity Assessment**

   Computes multiple planarity descriptors for molecular fragments and rings.

   **Planarity Metrics:**

   * **RMSD** - Root-mean-square deviation from best-fit plane
   * **Max deviation** - Maximum atomic displacement from plane
   * **Planarity score** - Normalized measure (0 = perfectly planar, 1 = highly non-planar)
   * **Thickness** - Distance between extreme atoms perpendicular to plane

   **Parameters:**

   * **coords** (:obj:`torch.Tensor`, shape (B, N, 3)) - Atomic coordinates  
   * **mask** (:obj:`torch.BoolTensor`, shape (B, N)) - Valid atom mask
   * **device** (:obj:`torch.device`) - Computation device

   **Returns:**

   * **planarity_rmsd** (:obj:`torch.Tensor`, shape (B,)) - RMSD values in Ångstroms
   * **planarity_max_dev** (:obj:`torch.Tensor`, shape (B,)) - Maximum deviations in Ångstroms  
   * **planarity_score** (:obj:`torch.Tensor`, shape (B,)) - Normalized planarity scores
   * **plane_normal** (:obj:`torch.Tensor`, shape (B, 3)) - Best-fit plane normals
   * **plane_centroid** (:obj:`torch.Tensor`, shape (B, 3)) - Plane centroids

   **Usage Example:**

   .. code-block:: python

      # Analyze planarity of aromatic rings
      ring_coords = extract_ring_coordinates(molecule)
      ring_mask = create_valid_atom_mask(ring_coords)
      
      rmsd, max_dev, score, normal, centroid = compute_planarity_metrics_batch(
          ring_coords, ring_mask, device
      )
      
      # Classify ring planarity
      for i, s in enumerate(score):
          if s < 0.1:
              print(f"Ring {i}: Highly planar (score: {s:.3f})")
          elif s < 0.3:
              print(f"Ring {i}: Moderately planar (score: {s:.3f})")
          else:
              print(f"Ring {i}: Non-planar (score: {s:.3f})")

Crystallographic Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: geometry_utils.compute_distances_to_crystallographic_planes_frac_batch

   **Special Plane Distance Analysis**

   Computes fractional distances from atoms to the 26 special crystallographic planes used in structure analysis.

   **Special Planes:**

   * **Primary planes**: (100), (010), (001) - unit cell faces
   * **Diagonal planes**: (110), (101), (011) and negatives - face diagonals  
   * **Body diagonal planes**: (111) and variants - space diagonals
   * **Multiple denominators**: 4 and 6 for enhanced precision

   **Parameters:**

   * **atom_frac_coords** (:obj:`torch.Tensor`, shape (B, A, 3)) - Fractional coordinates
   * **atom_mask** (:obj:`torch.BoolTensor`, shape (B, A)) - Valid atom indicators
   * **device** (:obj:`torch.device`) - Computation device

   **Returns:**

   * **torch.Tensor**, shape (B, A, 26) - Fractional distances to each special plane

   **Applications:**

   * Packing motif characterization
   * Symmetry analysis and space group validation
   * Crystal engineering and polymorph prediction
   * Structure factor analysis for powder diffraction

.. autofunction:: geometry_utils.compute_angles_between_bonds_and_crystallographic_planes_frac_batch

   **Bond-Plane Angular Analysis**

   Calculates angles between molecular bonds and crystallographic plane normals for orientation analysis.

   **Parameters:**

   * **atom_frac_coords** (:obj:`torch.Tensor`, shape (B, A, 3)) - Fractional atomic coordinates
   * **bond_atom1** (:obj:`torch.LongTensor`, shape (B, M)) - First bond atom indices
   * **bond_atom2** (:obj:`torch.LongTensor`, shape (B, M)) - Second bond atom indices  
   * **bond_mask** (:obj:`torch.BoolTensor`, shape (B, M)) - Valid bond indicators
   * **device** (:obj:`torch.device`) - Computation device

   **Returns:**

   * **torch.Tensor**, shape (B, M, 13) - Bond-plane angles in degrees

   **Interpretation:**

   * **0°** - Bond parallel to plane
   * **90°** - Bond perpendicular to plane  
   * **Intermediate values** - Various orientations

Order Parameter Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: geometry_utils.compute_global_steinhardt_order_parameters_batch

   **Global Steinhardt Q_l Order Parameters**

   Computes rotationally invariant order parameters that characterize local atomic environments and overall structural organization.

   **Mathematical Background:**

   Steinhardt order parameters are based on spherical harmonics expansion:

   .. math::

      Q_l = \sqrt{\frac{4\pi}{2l+1} \sum_{m=-l}^{l} |q_{lm}|^2}

   Where :math:`q_{lm}` are the averaged spherical harmonic coefficients.

   **Parameters:**

   * **atom_to_com_vecs** (:obj:`torch.Tensor`, shape (B, N, 3)) - Vectors from center of mass
   * **atom_mask** (:obj:`torch.BoolTensor`, shape (B, N)) - Valid atom indicators
   * **atom_weights** (:obj:`torch.Tensor` or None, shape (B, N)) - Optional atomic weights
   * **device** (:obj:`torch.device`) - Computation device
   * **l_values** (:obj:`List[int]`) - Order parameter degrees [2, 4, 6, 8, 10]
   * **eps** (:obj:`float`) - Numerical stability parameter

   **Returns:**

   * **torch.Tensor**, shape (B, len(l_values)) - Q_l values for each structure

   **Physical Interpretation:**

   * **Q_2** - Measures nematic ordering (molecular alignment)
   * **Q_4** - Detects tetrahedral vs. other local symmetries
   * **Q_6** - Sensitive to hexagonal/cubic ordering
   * **Q_8, Q_10** - Higher-order structural correlations

   **Applications:**

   * Phase transition characterization
   * Crystal quality assessment  
   * Polymorphism detection
   * Disorder quantification

Coordinate Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: geometry_utils.compute_atom_vectors_to_point_batch

   **Vector Computation to Reference Points**

   Calculates displacement vectors from atoms to specified reference points (centers of mass, centroids, etc.).

   **Parameters:**

   * **atom_coords** (:obj:`torch.Tensor`, shape (B, N, 3)) - Atomic coordinates
   * **atom_frac_coords** (:obj:`torch.Tensor`, shape (B, N, 3)) - Fractional coordinates
   * **atom_mask** (:obj:`torch.BoolTensor`, shape (B, N)) - Valid atom mask
   * **reference_coords** (:obj:`torch.Tensor`, shape (B, 3)) - Reference point coordinates
   * **reference_frac_coords** (:obj:`torch.Tensor`, shape (B, 3)) - Reference fractional coordinates
   * **device** (:obj:`torch.device`) - Computation device

   **Returns:**

   * **atom_to_ref_vec** (:obj:`torch.Tensor`, shape (B, N, 3)) - Cartesian displacement vectors
   * **atom_to_ref_frac_vec** (:obj:`torch.Tensor`, shape (B, N, 3)) - Fractional displacement vectors
   * **atom_to_ref_dist** (:obj:`torch.Tensor`, shape (B, N)) - Euclidean distances

.. autofunction:: geometry_utils.compute_quaternions_from_rotation_matrices

   **Rotation Matrix to Quaternion Conversion**

   Converts 3×3 rotation matrices to unit quaternions using robust numerical algorithms.

   **Parameters:**

   * **R** (:obj:`torch.Tensor`, shape (B, 3, 3)) - Proper rotation matrices (R^T R = I, det = +1)
   * **device** (:obj:`torch.device`) - Computation device

   **Returns:**

   * **torch.Tensor**, shape (B, 4) - Unit quaternions [w, x, y, z] with w ≥ 0

   **Algorithm Features:**

   * **Numerically stable** - Handles near-singular cases
   * **Consistent handedness** - Enforces w ≥ 0 convention
   * **Batch optimized** - Vectorized operations for efficiency

   **Applications:**

   * Molecular orientation analysis
   * Crystal symmetry operations
   * Rigid body motion decomposition
   * Interpolation between orientations

Fragment Pairwise Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: geometry_utils.compute_fragment_pairwise_vectors_and_distances_batch

   **Intra-Fragment Distance Matrix Computation**

   Calculates all pairwise distances within molecular fragments for shape analysis and internal geometry characterization.

   **Parameters:**

   * **atom_frac_coords** (:obj:`torch.Tensor`, shape (B, N, 3)) - Fractional atomic coordinates
   * **atom_mask** (:obj:`torch.BoolTensor`, shape (B, N)) - Valid atom indicators
   * **fragment_atom_assignments** (:obj:`torch.LongTensor`, shape (B, N)) - Fragment membership
   * **max_atoms_per_fragment** (:obj:`int`) - Maximum atoms for memory allocation
   * **device** (:obj:`torch.device`) - Computation device

   **Returns:**

   * **fragment_pairwise_vectors** (:obj:`torch.Tensor`) - Inter-atomic vectors within fragments
   * **fragment_pairwise_distances** (:obj:`torch.Tensor`) - Distance matrices for each fragment
   * **fragment_pairwise_mask** (:obj:`torch.BoolTensor`) - Valid pair indicators

   **Applications:**

   * Molecular flexibility analysis
   * Conformational change detection  
   * Internal coordinate validation
   * Shape descriptor computation

Performance Optimization
------------------------

**GPU Acceleration Guidelines**

The geometry_utils module is optimized for GPU computation with PyTorch. Follow these best practices:

.. code-block:: python

   # Optimal device usage
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
   # Batch size recommendations
   if device.type == 'cuda':
       batch_size = 64  # Larger batches for GPU
   else:
       batch_size = 16  # Smaller batches for CPU
   
   # Memory management
   torch.cuda.empty_cache()  # Clear cache between large computations
   
   # Data type optimization
   coords = coords.to(device, dtype=torch.float32)  # float32 sufficient for most cases

**Memory Usage Patterns**

* **Linear scaling**: Most functions scale O(B × N) with batch size and atom count
* **Quadratic scaling**: Pairwise functions scale O(B × N²) - use carefully
* **GPU memory**: Monitor usage with ``torch.cuda.memory_allocated()``

**Batch Size Optimization**

.. code-block:: python

   def optimal_batch_size(total_structures, available_memory_gb):
       """Estimate optimal batch size based on available GPU memory."""
       memory_per_structure_mb = 50  # Approximate for typical molecules
       structures_per_gb = 1000 / memory_per_structure_mb
       
       max_batch_size = int(available_memory_gb * structures_per_gb * 0.8)  # 80% safety margin
       return min(max_batch_size, total_structures)

Error Handling and Validation
------------------------------

**Common Error Patterns**

.. code-block:: python

   # Input validation
   def validate_geometry_inputs(coords, mask):
       if coords.dim() != 3:
           raise ValueError(f"Expected 3D coordinates tensor, got {coords.dim()}D")
       
       if coords.shape[:2] != mask.shape:
           raise ValueError("Coordinate and mask shapes must match")
       
       if torch.any(torch.isnan(coords)):
           raise ValueError("NaN values detected in coordinates")
       
       if torch.any(torch.isinf(coords)):
           raise ValueError("Infinite values detected in coordinates")

**Debugging Tools**

.. code-block:: python

   # Diagnostic functions
   def check_tensor_health(tensor, name):
       print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}")
       print(f"  Range: [{tensor.min():.3f}, {tensor.max():.3f}]")
       print(f"  NaN count: {torch.isnan(tensor).sum()}")
       print(f"  Inf count: {torch.isinf(tensor).sum()}")

Integration Examples
--------------------

**Comprehensive Molecular Analysis Pipeline**

.. code-block:: python

   import torch
   from geometry_utils import *

   def analyze_molecular_geometry(structures_batch):
       """Complete geometric analysis of molecular structures."""
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       results = {}
       
       # 1. Bond angle analysis
       angle_ids, angles, angle_mask, angle_indices = compute_bond_angles_batch(
           structures_batch['atom_labels'],
           structures_batch['coords'], 
           structures_batch['atom_mask'],
           structures_batch['bond_atom1'],
           structures_batch['bond_atom2'],
           structures_batch['bond_mask'],
           device
       )
       results['bond_angles'] = angles
       
       # 2. Torsion angle analysis  
       torsion_ids, torsions, torsion_mask, torsion_indices = compute_torsion_angles_batch(
           structures_batch['atom_labels'],
           structures_batch['coords'],
           structures_batch['atom_mask'],
           structures_batch['bond_atom1'],
           structures_batch['bond_atom2'], 
           structures_batch['bond_mask'],
           device
       )
       results['torsion_angles'] = torsions
       
       # 3. Planarity analysis for aromatic rings
       ring_planarity = compute_planarity_metrics_batch(
           structures_batch['ring_coords'],
           structures_batch['ring_mask'],
           device
       )
       results['planarity'] = ring_planarity
       
       # 4. Order parameter analysis
       order_params = compute_global_steinhardt_order_parameters_batch(
           structures_batch['atom_to_com_vectors'],
           structures_batch['atom_mask'],
           structures_batch['atom_weights'],
           device
       )
       results['order_parameters'] = order_params
       
       return results

**Crystallographic Structure Analysis**

.. code-block:: python

   def analyze_crystal_packing(crystal_batch):
       """Analyze crystallographic packing features."""
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
       # Distance to special planes
       plane_distances = compute_distances_to_crystallographic_planes_frac_batch(
           crystal_batch['frac_coords'],
           crystal_batch['atom_mask'],
           device
       )
       
       # Bond orientations relative to crystal axes
       bond_plane_angles = compute_angles_between_bonds_and_crystallographic_planes_frac_batch(
           crystal_batch['frac_coords'],
           crystal_batch['bond_atom1'],
           crystal_batch['bond_atom2'],
           crystal_batch['bond_mask'],
           device
       )
       
       return {
           'plane_distances': plane_distances,
           'bond_orientations': bond_plane_angles
       }

Cross-References
----------------

**Related CSA Modules:**

* :doc:`../processing/fragment_utils` - Fragment identification and properties
* :doc:`../processing/contact_utils` - Intermolecular contact analysis  
* :doc:`../processing/cell_utils` - Unit cell transformations
* :doc:`../extraction/structure_post_extraction_processor` - Main processing pipeline
* :doc:`../io/data_reader` - Input data handling

**External Dependencies:**

* `PyTorch <https://pytorch.org/>`_ - Tensor operations and GPU acceleration
* `NumPy <https://numpy.org/>`_ - Array operations and mathematical functions  
* `SciPy <https://scipy.org/>`_ - Advanced mathematical algorithms

**Mathematical References:**

* Steinhardt, P. J. et al. "Bond-orientational order in liquids and glasses" *Physical Review B* 28, 784 (1983)
* Allen, M. P. & Tildesley, D. J. "Computer Simulation of Liquids" Oxford University Press (2017)
* Giacovazzo, C. et al. "Fundamentals of Crystallography" Oxford University Press (2011)