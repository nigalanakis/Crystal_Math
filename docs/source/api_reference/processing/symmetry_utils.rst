symmetry_utils module
====================

.. automodule:: symmetry_utils
   :members:
   :undoc-members:
   :show-inheritance:

Crystallographic Symmetry Operations and Parsing
------------------------------------------------

The ``symmetry_utils`` module provides comprehensive tools for parsing, applying, and managing crystallographic symmetry operations. It handles the conversion of symmetry operator strings into mathematical transformations and their application to molecular contacts and hydrogen bonds.

**Key Features:**

* **Symmetry operator parsing** - Convert string representations to matrices
* **Matrix operations** - Rotation matrices and translation vectors
* **Inverse transformations** - Compute inverse symmetry operations
* **Batch processing** - Efficient GPU operations for large datasets
* **Contact expansion** - Apply symmetry to intermolecular interactions
* **Validation and debugging** - Robust error handling and diagnostics

Core Functions
--------------

.. autofunction:: symmetry_utils.parse_sym_op

   **Crystallographic Symmetry Operator Parsing**

   Parses International Tables symmetry operator strings into rotation matrices and translation vectors for mathematical operations.

   **Symmetry Operator Format:**

   Standard crystallographic notation uses comma-separated expressions for x, y, z coordinates:
   
   * **Identity**: ``'x, y, z'``
   * **Inversion**: ``'-x, -y, -z'``
   * **Translation**: ``'x+1/2, y+1/2, z'``
   * **Rotation + Translation**: ``'-y+1/2, x+1/2, z'``
   * **Complex operations**: ``'-x+1/3, -y+2/3, z+1/6'``

   **Parameters:**

   * **sym** (:obj:`str`) - Symmetry operator string in International Tables format

   **Returns:**

   * **A** (:obj:`torch.Tensor`, shape (3, 3)) - Integer rotation matrix
   * **t** (:obj:`torch.Tensor`, shape (3,)) - Float translation vector

   **Algorithm Details:**

   1. **Expression parsing** - Split by commas to get x, y, z transformations
   2. **Rotation extraction** - Identify coefficients of x, y, z variables
   3. **Translation extraction** - Parse constant terms (fractions and decimals)
   4. **Matrix construction** - Build 3×3 rotation matrix and 3-element translation vector

   **Raises:**

   * **ValueError** - If input string doesn't contain exactly three expressions

   **Usage Examples:**

   .. code-block:: python

      import torch
      from symmetry_utils import parse_sym_op

      # Identity operation
      A_id, t_id = parse_sym_op('x, y, z')
      print("Identity operation:")
      print(f"Rotation matrix:\n{A_id}")
      print(f"Translation vector: {t_id}")
      # Expected: 3×3 identity matrix, zero translation
      
      # Inversion center
      A_inv, t_inv = parse_sym_op('-x, -y, -z')
      print("Inversion operation:")
      print(f"Rotation matrix:\n{A_inv}")
      # Expected: -1×identity matrix
      
      # Two-fold screw axis along c
      A_screw, t_screw = parse_sym_op('-x, -y, z+1/2')
      print("Screw axis operation:")
      print(f"Rotation matrix:\n{A_screw}")
      print(f"Translation vector: {t_screw}")
      
      # Complex operation with fractions
      A_complex, t_complex = parse_sym_op('-y+1/3, x-y+2/3, z+1/6')
      print("Complex operation:")
      print(f"Rotation matrix:\n{A_complex}")
      print(f"Translation vector: {t_complex}")

   **Common Symmetry Operations:**

   .. code-block:: python

      # Standard crystallographic operations
      common_operations = {
          'identity': 'x, y, z',
          'inversion': '-x, -y, -z',
          'mirror_xy': 'x, y, -z',
          'mirror_xz': 'x, -y, z', 
          'mirror_yz': '-x, y, z',
          'twofold_x': 'x, -y, -z',
          'twofold_y': '-x, y, -z',
          'twofold_z': '-x, -y, z',
          'threefold_111': 'z, x, y',
          'fourfold_z': '-y, x, z',
          'sixfold_z': '-x+y, -x, z'
      }
      
      for name, operation in common_operations.items():
          A, t = parse_sym_op(operation)
          det = torch.det(A.float())
          print(f"{name}: det(A) = {det:.0f}")

.. autofunction:: symmetry_utils.invert_sym_op

   **Symmetry Operation Inversion**

   Computes the inverse of a crystallographic symmetry operation for reverse transformations.

   **Mathematical Foundation:**

   For a symmetry operation defined by rotation matrix **A** and translation vector **t**:

   .. math::

      \vec{r}' = \mathbf{A} \vec{r} + \vec{t}

   The inverse operation is:

   .. math::

      \vec{r} = \mathbf{A}^{-1} \vec{r}' - \mathbf{A}^{-1} \vec{t}

   Where :math:`\mathbf{A}^{-1} = \mathbf{A}^T` for orthogonal matrices.

   **Parameters:**

   * **A** (:obj:`torch.Tensor`, shape (3, 3)) - Rotation matrix
   * **t** (:obj:`torch.Tensor`, shape (3,)) - Translation vector

   **Returns:**

   * **A_inv** (:obj:`torch.Tensor`, shape (3, 3)) - Inverse rotation matrix
   * **t_inv** (:obj:`torch.Tensor`, shape (3,)) - Inverse translation vector

   **Raises:**

   * **ValueError** - If A is not shape (3,3) or t is not shape (3,)

   **Usage Examples:**

   .. code-block:: python

      # Test inverse operations
      original_op = 'x+1/2, -y+1/2, z'
      A, t = parse_sym_op(original_op)
      A_inv, t_inv = invert_sym_op(A, t)
      
      # Verify that applying operation then inverse gives identity
      test_point = torch.tensor([0.25, 0.75, 0.5])
      
      # Apply original operation
      transformed = A.float() @ test_point + t
      
      # Apply inverse operation
      recovered = A_inv.float() @ transformed + t_inv
      
      print(f"Original point: {test_point}")
      print(f"Transformed: {transformed}")
      print(f"Recovered: {recovered}")
      print(f"Difference: {torch.norm(recovered - test_point):.6f}")
      # Should be very close to zero

   **Symmetry Chain Verification:**

   .. code-block:: python

      def verify_symmetry_inversion(sym_op_string):
          """Verify that symmetry operation inversion is correct."""
          A, t = parse_sym_op(sym_op_string)
          A_inv, t_inv = invert_sym_op(A, t)
          
          # Test with random points
          test_points = torch.randn(10, 3)
          
          for point in test_points:
              # Forward transformation
              point_fwd = A.float() @ point + t
              
              # Inverse transformation
              point_back = A_inv.float() @ point_fwd + t_inv
              
              # Check if we recover original point
              error = torch.norm(point_back - point)
              if error > 1e-6:
                  print(f"Inversion failed for {sym_op_string}: error = {error}")
                  return False
          
          print(f"Inversion verified for {sym_op_string}")
          return True

Batch Symmetry Processing
-------------------------

.. autofunction:: symmetry_utils.add_symmetry_matrices

   **Batch Symmetry Matrix Addition to Parameter Dictionaries**

   Parses lists of symmetry operator strings and adds corresponding rotation and translation matrices to parameter dictionaries for batch processing.

   **Parameters:**

   * **parameters** (:obj:`Dict[str, Any]`) - Dictionary containing symmetry strings and coordinate tensors
   * **sym_key** (:obj:`str`) - Key for List[List[str]] of symmetry operator strings
   * **coords_key** (:obj:`str`) - Key for coordinate tensor to determine batch dimensions
   * **device** (:obj:`torch.device`, optional) - Target device for tensors

   **Modifies parameters dictionary by adding:**

   * **'{sym_key}_A'** (:obj:`torch.Tensor`, shape (B, N, 3, 3)) - Rotation matrices
   * **'{sym_key}_T'** (:obj:`torch.Tensor`, shape (B, N, 3)) - Translation vectors
   * **'{sym_key}_A_inv'** (:obj:`torch.Tensor`, shape (B, N, 3, 3)) - Inverse rotation matrices
   * **'{sym_key}_T_inv'** (:obj:`torch.Tensor`, shape (B, N, 3)) - Inverse translation vectors

   **Algorithm:**

   1. **Unique operation extraction** - Find all unique symmetry operators across batch
   2. **Batch parsing** - Parse each unique operator once for efficiency
   3. **Index mapping** - Create mapping from (batch, contact) to unique operations
   4. **Tensor construction** - Build batch tensors using advanced indexing
   5. **Inverse computation** - Calculate all inverse operations

   **Usage Example:**

   .. code-block:: python

      # Prepare parameter dictionary
      parameters = {
          'inter_cc_symmetry': [
              ['x, y, z', '-x, -y, z', 'x+1/2, -y, -z'],  # Structure 1
              ['x, y, z', '-x, y, -z']                     # Structure 2
          ],
          'inter_cc_central_atom_coords': torch.randn(2, 3, 3)  # (B=2, N=3, 3)
      }
      
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      
      # Add symmetry matrices
      add_symmetry_matrices(
          parameters, 
          'inter_cc_symmetry', 
          'inter_cc_central_atom_coords', 
          device
      )
      
      # Access results
      rotation_matrices = parameters['inter_cc_symmetry_A']  # (2, 3, 3, 3)
      translation_vectors = parameters['inter_cc_symmetry_T']  # (2, 3, 3)
      
      print(f"Rotation matrices shape: {rotation_matrices.shape}")
      print(f"Translation vectors shape: {translation_vectors.shape}")

Contact-Specific Symmetry Functions
-----------------------------------

.. autofunction:: symmetry_utils.add_inter_cc_symmetry

   **Add Intermolecular Contact Symmetry Matrices**

   Convenience function for adding symmetry matrices specifically for intermolecular contact analysis.

   **Parameters:**

   * **parameters** (:obj:`Dict[str, Any]`) - Must contain 'inter_cc_symmetry' and 'inter_cc_central_atom_coords'
   * **device** (:obj:`torch.device`, optional) - Computation device

   **Usage:**

   .. code-block:: python

      contact_params = {
          'inter_cc_symmetry': symmetry_operations,
          'inter_cc_central_atom_coords': central_coords,
          # ... other contact parameters
      }
      
      add_inter_cc_symmetry(contact_params, device)
      
      # Matrices are now available as:
      # contact_params['inter_cc_symmetry_A']
      # contact_params['inter_cc_symmetry_T']
      # contact_params['inter_cc_symmetry_A_inv']
      # contact_params['inter_cc_symmetry_T_inv']

.. autofunction:: symmetry_utils.add_inter_hb_symmetry

   **Add Hydrogen Bond Symmetry Matrices**

   Convenience function for adding symmetry matrices specifically for hydrogen bond analysis.

   **Parameters:**

   * **parameters** (:obj:`Dict[str, Any]`) - Must contain 'inter_hb_symmetry' and 'inter_hb_central_atom_coords'  
   * **device** (:obj:`torch.device`, optional) - Computation device

   **Applications in H-bond Analysis:**

   .. code-block:: python

      def analyze_hbond_symmetry_patterns(hbond_params):
          """Analyze symmetry patterns in hydrogen bonding."""
          
          # Add symmetry matrices
          add_inter_hb_symmetry(hbond_params)
          
          # Extract symmetry information
          symmetry_ops = hbond_params['inter_hb_symmetry']
          rotation_matrices = hbond_params['inter_hb_symmetry_A']
          
          # Analyze operation types
          operation_types = {}
          for batch_ops in symmetry_ops:
              for op in batch_ops:
                  op_type = classify_symmetry_operation(op)
                  operation_types[op_type] = operation_types.get(op_type, 0) + 1
          
          print("H-bond symmetry operation distribution:")
          for op_type, count in operation_types.items():
              print(f"  {op_type}: {count} operations")
          
          return operation_types

Advanced Symmetry Analysis
--------------------------

**Space Group Analysis**

.. code-block:: python

   def analyze_space_group_operations(symmetry_operations):
       """Analyze space group characteristics from symmetry operations."""
       
       # Parse all operations
       parsed_ops = []
       for op_string in symmetry_operations:
           A, t = parse_sym_op(op_string)
           parsed_ops.append((A, t, op_string))
       
       # Classify operation types
       operation_classes = {
           'identity': [],
           'inversion': [],
           'rotation': [],
           'reflection': [],
           'screw_axis': [],
           'glide_plane': []
       }
       
       for A, t, op_string in parsed_ops:
           det_A = torch.det(A.float())
           trace_A = torch.trace(A.float())
           has_translation = torch.norm(t) > 1e-6
           
           if torch.allclose(A.float(), torch.eye(3)) and not has_translation:
               operation_classes['identity'].append(op_string)
           elif det_A < 0:  # Improper rotation
               if has_translation:
                   operation_classes['glide_plane'].append(op_string)
               else:
                   operation_classes['reflection'].append(op_string)
           elif det_A > 0:  # Proper rotation
               if torch.allclose(A.float(), -torch.eye(3)):
                   operation_classes['inversion'].append(op_string)
               elif has_translation:
                   operation_classes['screw_axis'].append(op_string)
               else:
                   operation_classes['rotation'].append(op_string)
       
       return operation_classes

   def determine_rotation_order(rotation_matrix):
       """Determine the order of a rotation matrix."""
       A = rotation_matrix.float()
       
       # Apply rotation repeatedly until we get back to identity
       current = A.clone()
       for order in range(1, 13):  # Check up to 12-fold
           if torch.allclose(current, torch.eye(3), atol=1e-4):
               return order
           current = torch.matmul(current, A)
       
       return 1  # Default to identity

**Symmetry Element Detection**

.. code-block:: python

   def detect_symmetry_elements(symmetry_operations):
       """Detect and classify crystallographic symmetry elements."""
       
       elements = {
           'rotation_axes': {},
           'mirror_planes': [],
           'inversion_centers': [],
           'screw_axes': {},
           'glide_planes': []
       }
       
       for op_string in symmetry_operations:
           A, t = parse_sym_op(op_string)
           det_A = torch.det(A.float())
           has_translation = torch.norm(t) > 1e-6
           
           if det_A > 0:  # Proper rotation
               if not torch.allclose(A.float(), torch.eye(3)):
                   order = determine_rotation_order(A)
                   
                   if has_translation:
                       # Screw axis
                       key = f"{order}-fold"
                       if key not in elements['screw_axes']:
                           elements['screw_axes'][key] = []
                       elements['screw_axes'][key].append({
                           'operation': op_string,
                           'translation': t
                       })
                   else:
                       # Pure rotation
                       key = f"{order}-fold"
                       if key not in elements['rotation_axes']:
                           elements['rotation_axes'][key] = []
                       elements['rotation_axes'][key].append({
                           'operation': op_string
                       })
           
           elif det_A < 0:  # Improper rotation (mirror/inversion)
               if torch.allclose(A.float(), -torch.eye(3)):
                   elements['inversion_centers'].append(op_string)
               else:
                   if has_translation:
                       elements['glide_planes'].append({
                           'operation': op_string,
                           'translation': t
                       })
                   else:
                       elements['mirror_planes'].append({
                           'operation': op_string
                       })
       
       return elements

Performance Optimization
------------------------

**Efficient Symmetry Operation Caching**

.. code-block:: python

   class SymmetryOperationCache:
       """Cache for parsed symmetry operations to avoid repeated parsing."""
       
       def __init__(self):
           self.cache = {}
           self.inverse_cache = {}
       
       def parse_operation(self, op_string):
           """Parse operation with caching."""
           if op_string not in self.cache:
               A, t = parse_sym_op(op_string)
               A_inv, t_inv = invert_sym_op(A, t)
               
               self.cache[op_string] = (A, t)
               self.inverse_cache[op_string] = (A_inv, t_inv)
           
           return self.cache[op_string]
       
       def get_inverse(self, op_string):
           """Get inverse operation with caching."""
           if op_string not in self.inverse_cache:
               self.parse_operation(op_string)  # This will cache both
           
           return self.inverse_cache[op_string]
       
       def clear(self):
           """Clear cache to free memory."""
           self.cache.clear()
           self.inverse_cache.clear()

   # Global cache instance
   symmetry_cache = SymmetryOperationCache()

**Memory-Efficient Batch Processing**

.. code-block:: python

   def process_large_symmetry_batch(parameters, batch_size=1000):
       """Process large batches of symmetry operations efficiently."""
       
       total_size = len(parameters['inter_cc_symmetry'])
       device = parameters.get('device', torch.device('cpu'))
       
       # Process in chunks
       for start_idx in range(0, total_size, batch_size):
           end_idx = min(start_idx + batch_size, total_size)
           
           # Extract batch
           batch_params = {}
           for key, value in parameters.items():
               if isinstance(value, list):
                   batch_params[key] = value[start_idx:end_idx]
               elif isinstance(value, torch.Tensor):
                   batch_params[key] = value[start_idx:end_idx]
               else:
                   batch_params[key] = value
           
           # Process batch
           add_symmetry_matrices(
               batch_params,
               'inter_cc_symmetry',
               'inter_cc_central_atom_coords',
               device
           )
           
           # Store results (move to CPU to save GPU memory)
           for key in ['inter_cc_symmetry_A', 'inter_cc_symmetry_T',
                      'inter_cc_symmetry_A_inv', 'inter_cc_symmetry_T_inv']:
               if start_idx == 0:
                   parameters[key] = batch_params[key].cpu()
               else:
                   parameters[key] = torch.cat([
                       parameters[key], 
                       batch_params[key].cpu()
                   ], dim=0)
           
           # Clear GPU memory
           if device.type == 'cuda':
               torch.cuda.empty_cache()

Error Handling and Validation
------------------------------

**Symmetry Operation Validation**

.. code-block:: python

   def validate_symmetry_operation(op_string):
       """Validate a symmetry operation string."""
       
       try:
           A, t = parse_sym_op(op_string)
       except Exception as e:
           raise ValueError(f"Failed to parse symmetry operation '{op_string}': {e}")
       
       # Check that rotation matrix is valid
       det_A = torch.det(A.float())
       if not torch.allclose(torch.abs(det_A), torch.tensor(1.0), atol=1e-6):
           raise ValueError(f"Invalid rotation matrix determinant: {det_A}")
       
       # Check that matrix is integer
       if not torch.allclose(A.float(), A.float().round()):
           raise ValueError(f"Rotation matrix contains non-integer entries")
       
       # Check translation vector range
       if torch.any(torch.abs(t) >= 1.0):
           print(f"Warning: Translation vector has components >= 1.0")
       
       return True

   def validate_symmetry_matrices(parameters, sym_key):
       """Validate computed symmetry matrices."""
       
       A_key = f'{sym_key}_A'
       A_inv_key = f'{sym_key}_A_inv'
       
       if A_key not in parameters or A_inv_key not in parameters:
           raise ValueError("Symmetry matrices not found in parameters")
       
       A = parameters[A_key]
       A_inv = parameters[A_inv_key]
       
       # Check that A_inv is actually the inverse of A
       identity = torch.matmul(A, A_inv)
       expected_identity = torch.eye(3, device=A.device).expand_as(identity)
       
       if not torch.allclose(identity, expected_identity, atol=1e-4):
           raise ValueError("Computed inverse matrices are incorrect")

Integration Examples
--------------------

**Complete Symmetry Analysis Pipeline**

.. code-block:: python

   def comprehensive_symmetry_analysis(crystal_structures):
       """Perform complete symmetry analysis on crystal structures."""
       
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
       results = []
       for structure in crystal_structures:
           # Extract symmetry operations
           symmetry_ops = structure['symmetry_operations']
           
           # Validate operations
           valid_ops = []
           for op in symmetry_ops:
               try:
                   validate_symmetry_operation(op)
                   valid_ops.append(op)
               except ValueError as e:
                   print(f"Skipping invalid operation: {e}")
           
           # Analyze space group characteristics
           space_group_analysis = analyze_space_group_operations(valid_ops)
           symmetry_elements = detect_symmetry_elements(valid_ops)
           
           # Prepare for batch processing
           parameters = {
               'inter_cc_symmetry': [valid_ops],
               'inter_cc_central_atom_coords': structure['contact_coords'].unsqueeze(0)
           }
           
           # Add symmetry matrices
           add_inter_cc_symmetry(parameters, device)
           
           results.append({
               'structure_id': structure['id'],
               'valid_operations': valid_ops,
               'space_group_analysis': space_group_analysis,
               'symmetry_elements': symmetry_elements,
               'symmetry_matrices': {
                   'A': parameters['inter_cc_symmetry_A'],
                   'T': parameters['inter_cc_symmetry_T'],
                   'A_inv': parameters['inter_cc_symmetry_A_inv'],
                   'T_inv': parameters['inter_cc_symmetry_T_inv']
               }
           })
       
       return results

Cross-References
----------------

**Related CSA Modules:**

* :doc:`../processing/contact_utils` - Symmetry application to intermolecular contacts
* :doc:`../processing/cell_utils` - Unit cell transformations for symmetry operations
* :doc:`../processing/geometry_utils` - Coordinate transformations using symmetry
* :doc:`../extraction/structure_post_extraction_processor` - Symmetry integration in processing pipeline

**External Dependencies:**

* `PyTorch <https://pytorch.org/>`_ - Tensor operations and linear algebra
* `fractions` - Rational number parsing for crystallographic fractions

**Scientific References:**

* International Tables for Crystallography, Volume A: "Space-group symmetry" (2016)
* Shmueli, U. (ed.) "International Tables for Crystallography, Volume B" Kluwer Academic Publishers (2001)
* Aroyo, M. I. (ed.) "International Tables for Crystallography, Volume A1" Kluwer Academic Publishers (2006)
* Hahn, T. (ed.) "International Tables for Crystallography, Volume A" Kluwer Academic Publishers (2005)