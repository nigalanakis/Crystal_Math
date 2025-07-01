cell_utils module
================

.. automodule:: cell_utils
   :members:
   :undoc-members:
   :show-inheritance:

Crystallographic Unit Cell Transformations and Matrix Operations
----------------------------------------------------------------

The ``cell_utils`` module provides GPU-accelerated batch processing for crystallographic unit cell transformations, real-space matrix computations, and coordinate system conversions. It handles the mathematical operations required to convert between different representations of crystal structures.

**Key Features:**

* **Unit cell matrix construction** - Convert cell parameters to transformation matrices
* **Coordinate transformations** - Fractional ↔ Cartesian conversions
* **Cell parameter scaling** - Normalized cell parameter representations
* **Batch processing** - Efficient GPU operations for large structure datasets
* **Numerical stability** - Robust handling of edge cases and degenerate cells
* **Volume calculations** - Unit cell volumes and geometric properties

Core Functions
--------------

.. autofunction:: cell_utils.compute_cell_matrix_batch

   **Real-Space Unit Cell Matrix Construction**

   Converts crystallographic unit cell parameters (a, b, c, α, β, γ) into real-space transformation matrices for coordinate conversions and distance calculations.

   **Mathematical Foundation:**

   The transformation matrix converts fractional coordinates to Cartesian coordinates:

   .. math::

      \begin{pmatrix} x \\ y \\ z \end{pmatrix} = \mathbf{M} \begin{pmatrix} u \\ v \\ w \end{pmatrix}

   Where the transformation matrix **M** is constructed as:

   .. math::

      \mathbf{M} = \begin{pmatrix}
      a & 0 & 0 \\
      b \cos\gamma & b \sin\gamma & 0 \\
      c \cos\beta & c \frac{\cos\alpha - \cos\beta \cos\gamma}{\sin\gamma} & c_z
      \end{pmatrix}

   With :math:`c_z = \sqrt{c^2 - c^2\cos^2\beta - \left(c \frac{\cos\alpha - \cos\beta \cos\gamma}{\sin\gamma}\right)^2}`

   **Parameters:**

   * **lengths** (:obj:`torch.Tensor`, shape (B, 3)) - Unit cell lengths [a, b, c] in Ångstroms
   * **angles** (:obj:`torch.Tensor`, shape (B, 3)) - Unit cell angles [α, β, γ] in degrees
   * **device** (:obj:`torch.device`, optional) - Target computation device

   **Returns:**

   * **torch.Tensor**, shape (B, 3, 3) - Real-space cell transformation matrices

   **Physical Interpretation:**

   * **Column 1**: **a** vector along x-axis
   * **Column 2**: **b** vector in xy-plane 
   * **Column 3**: **c** vector completing right-handed system

   **Usage Examples:**

   .. code-block:: python

      import torch
      from cell_utils import compute_cell_matrix_batch

      # Cubic crystal system (simple case)
      lengths_cubic = torch.tensor([[5.0, 5.0, 5.0]])  # a=b=c=5Å
      angles_cubic = torch.tensor([[90.0, 90.0, 90.0]])  # α=β=γ=90°
      
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      
      cell_matrix_cubic = compute_cell_matrix_batch(lengths_cubic, angles_cubic, device)
      print("Cubic cell matrix:")
      print(cell_matrix_cubic[0])
      # Expected: 5×identity matrix
      
      # Triclinic crystal system (general case)
      lengths_triclinic = torch.tensor([[6.0, 8.0, 10.0]])
      angles_triclinic = torch.tensor([[75.0, 85.0, 95.0]])
      
      cell_matrix_triclinic = compute_cell_matrix_batch(lengths_triclinic, angles_triclinic, device)
      print("Triclinic cell matrix:")
      print(cell_matrix_triclinic[0])
      
      # Calculate unit cell volume using determinant
      volume = torch.det(cell_matrix_triclinic)
      print(f"Unit cell volume: {volume.item():.2f} Ų")

   **Coordinate Conversion Applications:**

   .. code-block:: python

      def fractional_to_cartesian(frac_coords, cell_matrix):
          """Convert fractional to Cartesian coordinates."""
          # frac_coords: (B, N, 3), cell_matrix: (B, 3, 3)
          return torch.matmul(frac_coords, cell_matrix.transpose(-2, -1))
      
      def cartesian_to_fractional(cart_coords, cell_matrix):
          """Convert Cartesian to fractional coordinates."""
          cell_matrix_inv = torch.inverse(cell_matrix)
          return torch.matmul(cart_coords, cell_matrix_inv.transpose(-2, -1))
      
      # Example usage
      frac_coords = torch.tensor([[[0.25, 0.25, 0.25]]])  # Fractional coordinates
      cart_coords = fractional_to_cartesian(frac_coords, cell_matrix_cubic)
      print(f"Cartesian coordinates: {cart_coords[0, 0]}")  # [1.25, 1.25, 1.25]

   **Crystal System Applications:**

   .. code-block:: python

      def analyze_crystal_system(lengths, angles, tolerance=1e-3):
          """Classify crystal system from unit cell parameters."""
          a, b, c = lengths.unbind(dim=-1)
          alpha, beta, gamma = angles.unbind(dim=-1)
          
          # Check for cubic
          cubic = (torch.abs(a - b) < tolerance) & (torch.abs(b - c) < tolerance) & \
                  (torch.abs(alpha - 90) < tolerance) & (torch.abs(beta - 90) < tolerance) & \
                  (torch.abs(gamma - 90) < tolerance)
          
          # Check for tetragonal  
          tetragonal = (torch.abs(a - b) < tolerance) & (torch.abs(c - a) > tolerance) & \
                       (torch.abs(alpha - 90) < tolerance) & (torch.abs(beta - 90) < tolerance) & \
                       (torch.abs(gamma - 90) < tolerance)
          
          # Check for hexagonal
          hexagonal = (torch.abs(a - b) < tolerance) & (torch.abs(c - a) > tolerance) & \
                      (torch.abs(alpha - 90) < tolerance) & (torch.abs(beta - 90) < tolerance) & \
                      (torch.abs(gamma - 120) < tolerance)
          
          # Classify
          systems = []
          for i in range(len(lengths)):
              if cubic[i]:
                  systems.append("cubic")
              elif tetragonal[i]:
                  systems.append("tetragonal")
              elif hexagonal[i]:
                  systems.append("hexagonal")
              else:
                  systems.append("triclinic")
          
          return systems

   **Distance Calculations:**

   .. code-block:: python

      def compute_distances_with_pbc(coord1, coord2, cell_matrix, use_minimum_image=True):
          """Compute distances with periodic boundary conditions."""
          
          # Convert to fractional coordinates for PBC
          cell_inv = torch.inverse(cell_matrix)
          frac1 = torch.matmul(coord1, cell_inv.transpose(-2, -1))
          frac2 = torch.matmul(coord2, cell_inv.transpose(-2, -1))
          
          # Compute difference vector
          diff_frac = frac2 - frac1
          
          if use_minimum_image:
              # Apply minimum image convention
              diff_frac = diff_frac - torch.round(diff_frac)
          
          # Convert back to Cartesian
          diff_cart = torch.matmul(diff_frac, cell_matrix.transpose(-2, -1))
          
          # Calculate distance
          distances = torch.norm(diff_cart, dim=-1)
          return distances

.. autofunction:: cell_utils.compute_scaled_cell

   **Normalized Cell Parameter Representation**

   Computes scaled unit cell parameters for machine learning applications and comparative analysis by normalizing lengths and angles.

   **Scaling Scheme:**

   The scaling transforms unit cell parameters into a dimensionless 6-component feature vector:

   .. math::

      \vec{f}_{cell} = [1.0, \frac{b}{a}, \frac{c}{a}, \frac{\alpha}{90°}, \frac{\beta}{90°}, \frac{\gamma}{90°}]

   **Parameters:**

   * **lengths** (:obj:`torch.Tensor`, shape (B, 3)) - Unit cell lengths [a, b, c]
   * **angles** (:obj:`torch.Tensor`, shape (B, 3)) - Unit cell angles [α, β, γ] in degrees
   * **device** (:obj:`torch.device`, optional) - Target computation device

   **Returns:**

   * **torch.Tensor**, shape (B, 6) - Scaled cell feature vectors

   **Advantages of Scaling:**

   * **Dimensionless representation** - Removes length scale dependence
   * **Machine learning compatibility** - Normalized features for ML models
   * **Comparative analysis** - Enables cross-dataset comparisons
   * **Reduced parameter space** - From 6 to 5 independent dimensions

   **Usage Examples:**

   .. code-block:: python

      # Compare different crystal structures
      structures = {
          'diamond': {'lengths': [3.57, 3.57, 3.57], 'angles': [90, 90, 90]},
          'graphite': {'lengths': [2.46, 2.46, 6.71], 'angles': [90, 90, 120]},
          'quartz': {'lengths': [4.91, 4.91, 5.41], 'angles': [90, 90, 120]}
      }
      
      for name, params in structures.items():
          lengths = torch.tensor([params['lengths']])
          angles = torch.tensor([params['angles']])
          scaled = compute_scaled_cell(lengths, angles)
          
          print(f"{name}: {scaled[0].tolist()}")
          
      # Use for machine learning features
      def extract_cell_features(crystal_dataset):
          """Extract scaled cell features for ML."""
          features = []
          for structure in crystal_dataset:
              scaled = compute_scaled_cell(
                  structure['cell_lengths'], 
                  structure['cell_angles']
              )
              features.append(scaled)
          return torch.cat(features, dim=0)

   **Statistical Analysis Applications:**

   .. code-block:: python

      def analyze_cell_distributions(scaled_cells):
          """Analyze statistical distributions of scaled cell parameters."""
          
          # Component labels
          labels = ['1.0', 'b/a', 'c/a', 'α/90°', 'β/90°', 'γ/90°']
          
          # Compute statistics
          means = scaled_cells.mean(dim=0)
          stds = scaled_cells.std(dim=0)
          
          print("Cell parameter statistics:")
          for i, label in enumerate(labels):
              print(f"  {label}: {means[i]:.3f} ± {stds[i]:.3f}")
          
          # Identify outliers
          z_scores = torch.abs((scaled_cells - means) / stds)
          outliers = z_scores > 3.0  # 3-sigma outliers
          
          outlier_structures = torch.any(outliers, dim=1)
          print(f"Outlier structures: {outlier_structures.sum().item()}")
          
          return {
              'means': means,
              'stds': stds,
              'outliers': outlier_structures
          }

Advanced Cell Operations
------------------------

**Volume and Density Calculations**

.. code-block:: python

   def compute_cell_volumes(cell_matrices):
       """Compute unit cell volumes from transformation matrices."""
       # Volume is the determinant of the cell matrix
       volumes = torch.det(cell_matrices)
       return torch.abs(volumes)  # Ensure positive volumes

   def compute_cell_densities(cell_matrices, molecular_weights, z_values):
       """Compute crystal densities from cell parameters."""
       volumes = compute_cell_volumes(cell_matrices)  # Ų
       
       # Convert to cm³ and compute density
       avogadro = 6.022e23
       volumes_cm3 = volumes * 1e-24  # Ų to cm³
       
       # Density = (Z × MW) / (V × N_A)
       densities = (z_values * molecular_weights) / (volumes_cm3 * avogadro)
       return densities  # g/cm³

   def compute_reciprocal_lattice(cell_matrices):
       """Compute reciprocal lattice vectors."""
       # Reciprocal lattice is 2π × (direct lattice)^(-T)
       volumes = compute_cell_volumes(cell_matrices)
       
       # B = 2π × A^(-T)
       reciprocal = 2 * torch.pi * torch.inverse(cell_matrices).transpose(-2, -1)
       reciprocal_volumes = (2 * torch.pi)**3 / volumes
       
       return reciprocal, reciprocal_volumes

**Metric Tensor Operations**

.. code-block:: python

   def compute_metric_tensor(cell_matrices):
       """Compute metric tensor for distance calculations."""
       # G = A^T × A
       metric = torch.matmul(cell_matrices.transpose(-2, -1), cell_matrices)
       return metric

   def compute_distance_squared_fractional(frac_coords1, frac_coords2, metric_tensor):
       """Compute squared distances in fractional coordinates using metric tensor."""
       diff = frac_coords2 - frac_coords1
       
       # d² = Δu^T × G × Δu
       dist_sq = torch.sum(diff.unsqueeze(-2) @ metric_tensor @ diff.unsqueeze(-1), dim=(-2, -1))
       return dist_sq.squeeze()

**Crystal System Classification**

.. code-block:: python

   def classify_crystal_systems_batch(lengths, angles, tolerance=1.0):
       """Classify crystal systems for a batch of structures."""
       a, b, c = lengths.unbind(dim=-1)
       alpha, beta, gamma = angles.unbind(dim=-1)
       
       # Define classification criteria
       same_ab = torch.abs(a - b) < tolerance
       same_bc = torch.abs(b - c) < tolerance  
       same_ac = torch.abs(a - c) < tolerance
       
       angle_90 = lambda x: torch.abs(x - 90) < tolerance
       angle_120 = lambda x: torch.abs(x - 120) < tolerance
       
       all_90 = angle_90(alpha) & angle_90(beta) & angle_90(gamma)
       alpha_beta_90 = angle_90(alpha) & angle_90(beta)
       gamma_120 = angle_120(gamma)
       
       # Classification logic
       cubic = same_ab & same_bc & all_90
       tetragonal = same_ab & ~same_bc & all_90
       orthorhombic = ~same_ab & ~same_bc & ~same_ac & all_90
       hexagonal = same_ab & ~same_bc & alpha_beta_90 & gamma_120
       trigonal = same_ab & same_bc & ~all_90  # Simplified
       monoclinic = ~all_90 & (angle_90(alpha) & angle_90(gamma))  # β ≠ 90°
       triclinic = ~all_90 & ~monoclinic
       
       # Convert to categorical labels
       systems = torch.zeros(lengths.shape[0], dtype=torch.long)
       systems[cubic] = 0      # Cubic
       systems[tetragonal] = 1 # Tetragonal  
       systems[orthorhombic] = 2  # Orthorhombic
       systems[hexagonal] = 3  # Hexagonal
       systems[trigonal] = 4   # Trigonal
       systems[monoclinic] = 5 # Monoclinic
       systems[triclinic] = 6  # Triclinic
       
       return systems

Performance Optimization
------------------------

**Memory-Efficient Matrix Operations**

.. code-block:: python

   def batch_cell_operations_memory_efficient(lengths, angles, batch_size=1000):
       """Process large datasets with memory management."""
       n_structures = lengths.shape[0]
       device = lengths.device
       
       cell_matrices = []
       scaled_cells = []
       
       for i in range(0, n_structures, batch_size):
           end_idx = min(i + batch_size, n_structures)
           batch_lengths = lengths[i:end_idx]
           batch_angles = angles[i:end_idx]
           
           # Process batch
           batch_matrices = compute_cell_matrix_batch(batch_lengths, batch_angles, device)
           batch_scaled = compute_scaled_cell(batch_lengths, batch_angles, device)
           
           # Move to CPU to save GPU memory
           cell_matrices.append(batch_matrices.cpu())
           scaled_cells.append(batch_scaled.cpu())
           
           # Clear GPU cache
           if device.type == 'cuda':
               torch.cuda.empty_cache()
       
       return torch.cat(cell_matrices), torch.cat(scaled_cells)

**Vectorized Distance Calculations**

.. code-block:: python

   def compute_pairwise_distances_efficient(coords, cell_matrices, max_distance=15.0):
       """Efficiently compute pairwise distances with distance cutoff."""
       B, N, _ = coords.shape
       device = coords.device
       
       # Convert to fractional coordinates
       cell_inv = torch.inverse(cell_matrices)
       frac_coords = torch.matmul(coords, cell_inv.transpose(-2, -1))
       
       # Compute all pairwise differences
       diff = frac_coords.unsqueeze(2) - frac_coords.unsqueeze(1)  # (B, N, N, 3)
       
       # Apply minimum image convention
       diff = diff - torch.round(diff)
       
       # Convert back to Cartesian for distance calculation
       diff_cart = torch.matmul(diff, cell_matrices.unsqueeze(1))  # Broadcasting
       distances = torch.norm(diff_cart, dim=-1)
       
       # Apply distance cutoff
       mask = distances < max_distance
       distances = distances * mask.float()
       
       return distances, mask

Error Handling and Validation
------------------------------

**Cell Parameter Validation**

.. code-block:: python

   def validate_cell_parameters(lengths, angles):
       """Validate crystallographic cell parameters."""
       
       # Check for positive lengths
       if torch.any(lengths <= 0):
           raise ValueError("Cell lengths must be positive")
       
       # Check angle ranges
       if torch.any((angles <= 0) | (angles >= 180)):
           raise ValueError("Cell angles must be between 0° and 180°")
       
       # Check for physically reasonable values
       if torch.any(lengths > 1000):  # Unreasonably large cell
           raise ValueError("Cell lengths exceed reasonable limits (>1000 Å)")
       
       if torch.any(lengths < 1):  # Unreasonably small cell
           raise ValueError("Cell lengths below reasonable limits (<1 Å)")
       
       # Check for degenerate cases
       for i in range(len(lengths)):
           a, b, c = lengths[i]
           alpha, beta, gamma = angles[i]
           
           # Triangle inequality checks for cell validity
           cos_alpha = torch.cos(alpha * torch.pi / 180)
           cos_beta = torch.cos(beta * torch.pi / 180)
           cos_gamma = torch.cos(gamma * torch.pi / 180)
           
           # Validate that the cell can be constructed
           discriminant = 1 + 2*cos_alpha*cos_beta*cos_gamma - cos_alpha**2 - cos_beta**2 - cos_gamma**2
           
           if discriminant <= 0:
               raise ValueError(f"Invalid cell parameters for structure {i}: cannot construct valid unit cell")

**Debugging and Diagnostics**

.. code-block:: python

   def diagnose_cell_matrices(cell_matrices, expected_volumes=None):
       """Diagnose cell matrix computations for quality control."""
       
       print("Cell Matrix Diagnostics:")
       
       # Check for valid matrices
       determinants = torch.det(cell_matrices)
       volumes = torch.abs(determinants)
       
       print(f"  Volume range: {volumes.min():.2f} - {volumes.max():.2f} Ų")
       print(f"  Mean volume: {volumes.mean():.2f} ± {volumes.std():.2f} Ų")
       
       # Check for negative determinants (left-handed systems)
       negative_det = determinants < 0
       if negative_det.any():
           print(f"  Warning: {negative_det.sum()} structures have negative determinants")
       
       # Check for degenerate cells (zero volume)
       zero_volume = volumes < 1e-6
       if zero_volume.any():
           print(f"  Error: {zero_volume.sum()} structures have near-zero volumes")
       
       # Check orthogonality
       for i in range(min(5, len(cell_matrices))):  # Check first 5 structures
           M = cell_matrices[i]
           
           # Compute angles between vectors
           a_vec = M[0]
           b_vec = M[1] 
           c_vec = M[2]
           
           ab_angle = torch.acos(torch.dot(a_vec, b_vec) / (torch.norm(a_vec) * torch.norm(b_vec)))
           ac_angle = torch.acos(torch.dot(a_vec, c_vec) / (torch.norm(a_vec) * torch.norm(c_vec)))
           bc_angle = torch.acos(torch.dot(b_vec, c_vec) / (torch.norm(b_vec) * torch.norm(c_vec)))
           
           ab_deg = ab_angle * 180 / torch.pi
           ac_deg = ac_angle * 180 / torch.pi
           bc_deg = bc_angle * 180 / torch.pi
           
           print(f"  Structure {i} vector angles: {ab_deg:.1f}°, {ac_deg:.1f}°, {bc_deg:.1f}°")

Integration Examples
--------------------

**Complete Crystal Analysis Pipeline**

.. code-block:: python

   def comprehensive_cell_analysis(crystal_dataset):
       """Perform complete crystallographic cell analysis."""
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
       # Extract cell parameters
       lengths = torch.stack([s['cell_lengths'] for s in crystal_dataset])
       angles = torch.stack([s['cell_angles'] for s in crystal_dataset])
       
       # Validate parameters
       validate_cell_parameters(lengths, angles)
       
       # Compute transformations
       cell_matrices = compute_cell_matrix_batch(lengths, angles, device)
       scaled_cells = compute_scaled_cell(lengths, angles, device)
       
       # Compute derived properties
       volumes = compute_cell_volumes(cell_matrices)
       crystal_systems = classify_crystal_systems_batch(lengths, angles)
       
       # Analyze distributions
       stats = analyze_cell_distributions(scaled_cells)
       
       return {
           'cell_matrices': cell_matrices,
           'scaled_cells': scaled_cells,
           'volumes': volumes,
           'crystal_systems': crystal_systems,
           'statistics': stats
       }

Cross-References
----------------

**Related CSA Modules:**

* :doc:`../processing/geometry_utils` - Coordinate transformations and distance calculations
* :doc:`../processing/contact_utils` - Intermolecular distance computations
* :doc:`../processing/symmetry_utils` - Symmetry operation applications
* :doc:`../extraction/structure_post_extraction_processor` - Cell matrix usage in processing pipeline

**External Dependencies:**

* `PyTorch <https://pytorch.org/>`_ - Tensor operations and GPU acceleration
* `NumPy <https://numpy.org/>`_ - Mathematical functions and array operations

**Scientific References:**

* Giacovazzo, C. et al. "Fundamentals of Crystallography" Oxford University Press (2011)
* Massa, W. "Crystal Structure Determination" Springer (2004)
* Shmueli, U. (ed.) "International Tables for Crystallography, Volume B" Kluwer Academic Publishers (2001)