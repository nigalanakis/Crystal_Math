contact_utils module
===================

.. automodule:: contact_utils
   :members:
   :undoc-members:
   :show-inheritance:

Intermolecular Contact Analysis and Symmetry Expansion
------------------------------------------------------

The ``contact_utils`` module provides GPU-accelerated batch processing for analyzing intermolecular contacts and hydrogen bonds in crystallographic structures. It handles symmetry expansion, contact classification, and fragment-level contact mapping essential for crystal packing analysis.

**Key Features:**

* **Symmetry expansion** - Apply crystallographic symmetry operations to contacts
* **Hydrogen bond detection** - Geometric criteria for H-bond identification
* **Contact classification** - van der Waals, close contacts, π-π interactions
* **Fragment mapping** - Map atomic contacts to molecular fragment interactions
* **Batch processing** - Efficient GPU-accelerated operations for large datasets
* **Distance calculations** - Accurate intermolecular distance computations

Core Contact Functions
----------------------

.. autofunction:: contact_utils.compute_symmetric_contacts_batch

   **Symmetry-Expanded Intermolecular Contact Analysis**

   Expands intermolecular contacts by applying precomputed crystallographic symmetry operations to generate the complete set of contacts in the crystal structure.

   **Symmetry Expansion Process:**

   1. **Parse symmetry operators** - Extract rotation matrices and translation vectors
   2. **Apply transformations** - Transform contact atom coordinates using symmetry
   3. **Calculate distances** - Compute actual intermolecular distances in 3D space
   4. **Validate contacts** - Filter contacts within specified distance cutoffs
   5. **Generate reciprocal contacts** - Create bidirectional contact pairs

   **Parameters:**

   * **central_atom_label** (:obj:`List[List[str]]`) - Central atom labels per structure
   * **contact_atom_label** (:obj:`List[List[str]]`) - Contact atom labels per structure
   * **central_atom_idx** (:obj:`torch.LongTensor`, shape (B, C)) - Central atom indices
   * **contact_atom_idx** (:obj:`torch.LongTensor`, shape (B, C)) - Contact atom indices
   * **central_atom_frac_coords** (:obj:`torch.Tensor`, shape (B, C, 3)) - Central atom fractional coordinates
   * **contact_atom_frac_coords** (:obj:`torch.Tensor`, shape (B, C, 3)) - Contact atom fractional coordinates
   * **lengths** (:obj:`torch.Tensor`, shape (B, C)) - Original contact distances
   * **strengths** (:obj:`torch.Tensor`, shape (B, C)) - Contact strength metrics
   * **in_los** (:obj:`torch.Tensor`, shape (B, C)) - Line-of-sight contact flags
   * **symmetry_A** (:obj:`torch.Tensor`, shape (B, C, 3, 3)) - Symmetry rotation matrices
   * **symmetry_T** (:obj:`torch.Tensor`, shape (B, C, 3)) - Symmetry translation vectors
   * **symmetry_A_inv** (:obj:`torch.Tensor`, shape (B, C, 3, 3)) - Inverse rotation matrices
   * **symmetry_T_inv** (:obj:`torch.Tensor`, shape (B, C, 3)) - Inverse translation vectors
   * **cell_matrix** (:obj:`torch.Tensor`, shape (B, 3, 3)) - Unit cell transformation matrices
   * **device** (:obj:`torch.device`) - GPU/CPU computation device

   **Returns:**

   Tuple containing expanded contact data:

   * **central_atom_labels** (:obj:`List[List[str]]`) - Expanded central atom labels
   * **contact_atom_labels** (:obj:`List[List[str]]`) - Expanded contact atom labels
   * **central_atom_idx** (:obj:`torch.Tensor`) - Expanded central atom indices
   * **contact_atom_idx** (:obj:`torch.Tensor`) - Expanded contact atom indices
   * **central_atom_coords** (:obj:`torch.Tensor`) - Expanded central atom Cartesian coordinates
   * **contact_atom_coords** (:obj:`torch.Tensor`) - Expanded contact atom Cartesian coordinates
   * **central_atom_frac_coords** (:obj:`torch.Tensor`) - Expanded fractional coordinates
   * **contact_atom_frac_coords** (:obj:`torch.Tensor`) - Expanded fractional coordinates
   * **lengths** (:obj:`torch.Tensor`) - Recalculated contact distances
   * **strengths** (:obj:`torch.Tensor`) - Contact strength values
   * **in_los** (:obj:`torch.Tensor`) - Line-of-sight flags
   * **symmetry_A** (:obj:`torch.Tensor`) - Applied rotation matrices
   * **symmetry_T** (:obj:`torch.Tensor`) - Applied translation vectors
   * **symmetry_A_inv** (:obj:`torch.Tensor`) - Inverse rotation matrices
   * **symmetry_T_inv** (:obj:`torch.Tensor`) - Inverse translation vectors
   * **contact_mask** (:obj:`torch.BoolTensor`) - Valid contact indicators

   **Mathematical Implementation:**

   Symmetry transformation is applied as:

   .. math::

      \vec{r}_{contact}^{sym} = \mathbf{A} \cdot \vec{r}_{contact}^{frac} + \vec{t}

   Distance calculation in Cartesian space:

   .. math::

      d = |\mathbf{M} \cdot (\vec{r}_{contact}^{sym} - \vec{r}_{central}^{frac})|

   Where :math:`\mathbf{M}` is the unit cell matrix.

   **Usage Example:**

   .. code-block:: python

      import torch
      from contact_utils import compute_symmetric_contacts_batch

      # Sample contact data for a benzene crystal
      central_labels = [['C1', 'C2']]
      contact_labels = [['C1_sym', 'C2_sym']]
      
      central_idx = torch.tensor([[0, 1]])
      contact_idx = torch.tensor([[6, 7]])  # Atoms in neighboring molecules
      
      # Fractional coordinates
      central_frac = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]])
      contact_frac = torch.tensor([[[0.5, 0.5, 0.0], [0.6, 0.5, 0.0]]])
      
      # Contact properties
      lengths = torch.tensor([[3.5, 3.8]])  # Å
      strengths = torch.tensor([[1.0, 0.8]])
      in_los = torch.tensor([[True, True]])
      
      # Symmetry operations (identity for this example)
      B, C = 1, 2
      symmetry_A = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, C, 3, 3)
      symmetry_T = torch.zeros(B, C, 3)
      
      # Unit cell matrix
      cell_matrix = torch.tensor([[[10.0, 0.0, 0.0],
                                   [0.0, 8.0, 0.0], 
                                   [0.0, 0.0, 6.0]]])
      
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      
      expanded_contacts = compute_symmetric_contacts_batch(
          central_labels, contact_labels,
          central_idx, contact_idx,
          central_frac, contact_frac,
          lengths, strengths, in_los,
          symmetry_A, symmetry_T, symmetry_A, symmetry_T,
          cell_matrix, device
      )
      
      print(f"Original contacts: {C}")
      print(f"Expanded contacts: {expanded_contacts[-1].sum().item()}")

Hydrogen Bond Analysis
----------------------

.. autofunction:: contact_utils.compute_symmetric_hbonds_batch

   **Hydrogen Bond Symmetry Expansion and Analysis**

   Expands hydrogen bonds using crystallographic symmetry operations with specialized handling for three-atom D-H⋯A interactions.

   **Hydrogen Bond Criteria:**

   * **Distance criterion**: D⋯A distance < 3.5 Å (typically)
   * **Angle criterion**: D-H⋯A angle > 120° (typically)
   * **Chemical criterion**: D = N, O, S; A = N, O, F, Cl
   * **Geometric optimization**: H⋯A distance minimized

   **Parameters:**

   * **central_atom_label** (:obj:`List[List[str]]`) - Donor atom labels (D)
   * **hydrogen_atom_label** (:obj:`List[List[str]]`) - Hydrogen atom labels (H)
   * **contact_atom_label** (:obj:`List[List[str]]`) - Acceptor atom labels (A)
   * **central_atom_idx** (:obj:`torch.LongTensor`, shape (B, H)) - Donor atom indices
   * **hydrogen_atom_idx** (:obj:`torch.LongTensor`, shape (B, H)) - Hydrogen atom indices
   * **contact_atom_idx** (:obj:`torch.LongTensor`, shape (B, H)) - Acceptor atom indices
   * **central_atom_frac_coords** (:obj:`torch.Tensor`, shape (B, H, 3)) - Donor fractional coordinates
   * **hydrogen_atom_frac_coords** (:obj:`torch.Tensor`, shape (B, H, 3)) - Hydrogen fractional coordinates
   * **contact_atom_frac_coords** (:obj:`torch.Tensor`, shape (B, H, 3)) - Acceptor fractional coordinates
   * **lengths** (:obj:`torch.Tensor`, shape (B, H)) - D⋯A distances
   * **angles** (:obj:`torch.Tensor`, shape (B, H)) - D-H⋯A angles in degrees
   * **in_los** (:obj:`torch.Tensor`, shape (B, H)) - Line-of-sight flags
   * **symmetry_A** (:obj:`torch.Tensor`, shape (B, H, 3, 3)) - Rotation matrices
   * **symmetry_T** (:obj:`torch.Tensor`, shape (B, H, 3)) - Translation vectors
   * **symmetry_A_inv** (:obj:`torch.Tensor`, shape (B, H, 3, 3)) - Inverse rotations
   * **symmetry_T_inv** (:obj:`torch.Tensor`, shape (B, H, 3)) - Inverse translations
   * **cell_matrix** (:obj:`torch.Tensor`, shape (B, 3, 3)) - Unit cell matrices
   * **device** (:obj:`torch.device`) - Computation device

   **Returns:**

   Tuple containing expanded hydrogen bond data:

   * **central_atom_labels** (:obj:`List[List[str]]`) - Expanded donor labels
   * **hydrogen_atom_labels** (:obj:`List[List[str]]`) - Expanded hydrogen labels
   * **contact_atom_labels** (:obj:`List[List[str]]`) - Expanded acceptor labels
   * **central_atom_idx** (:obj:`torch.Tensor`) - Expanded donor indices
   * **hydrogen_atom_idx** (:obj:`torch.Tensor`) - Expanded hydrogen indices
   * **contact_atom_idx** (:obj:`torch.Tensor`) - Expanded acceptor indices
   * **central_atom_coords** (:obj:`torch.Tensor`) - Donor Cartesian coordinates
   * **hydrogen_atom_coords** (:obj:`torch.Tensor`) - Hydrogen Cartesian coordinates
   * **contact_atom_coords** (:obj:`torch.Tensor`) - Acceptor Cartesian coordinates
   * **central_atom_frac_coords** (:obj:`torch.Tensor`) - Donor fractional coordinates
   * **hydrogen_atom_frac_coords** (:obj:`torch.Tensor`) - Hydrogen fractional coordinates
   * **contact_atom_frac_coords** (:obj:`torch.Tensor`) - Acceptor fractional coordinates
   * **lengths** (:obj:`torch.Tensor`) - Recalculated D⋯A distances
   * **angles** (:obj:`torch.Tensor`) - Recalculated D-H⋯A angles
   * **in_los** (:obj:`torch.Tensor`) - Line-of-sight flags
   * **symmetry_A** (:obj:`torch.Tensor`) - Applied rotation matrices
   * **symmetry_T** (:obj:`torch.Tensor`) - Applied translation vectors
   * **symmetry_A_inv** (:obj:`torch.Tensor`) - Inverse rotation matrices
   * **symmetry_T_inv** (:obj:`torch.Tensor`) - Inverse translation vectors
   * **hbond_mask** (:obj:`torch.BoolTensor`) - Valid hydrogen bond indicators

   **Hydrogen Bond Classification:**

   .. code-block:: python

      def classify_hydrogen_bonds(lengths, angles, donor_types, acceptor_types):
          """Classify hydrogen bonds by strength and type."""
          
          # Strength classification
          strong_hbonds = (lengths < 2.5) & (angles > 160)  # Strong H-bonds
          moderate_hbonds = (lengths < 3.0) & (angles > 140)  # Moderate H-bonds
          weak_hbonds = (lengths < 3.5) & (angles > 120)  # Weak H-bonds
          
          # Type classification
          oh_o = (donor_types == 'OH') & (acceptor_types == 'O')  # O-H⋯O
          nh_o = (donor_types == 'NH') & (acceptor_types == 'O')  # N-H⋯O
          oh_n = (donor_types == 'OH') & (acceptor_types == 'N')  # O-H⋯N
          
          return {
              'strong': strong_hbonds,
              'moderate': moderate_hbonds, 
              'weak': weak_hbonds,
              'OH_O': oh_o,
              'NH_O': nh_o,
              'OH_N': oh_n
          }

   **Usage Example:**

   .. code-block:: python

      # Analyze hydrogen bonding in ice crystal
      donor_labels = [['O1', 'O2']]
      hydrogen_labels = [['H1', 'H2']]
      acceptor_labels = [['O3', 'O4']]
      
      # Typical ice H-bond geometry
      lengths = torch.tensor([[2.8, 2.9]])  # O⋯O distances
      angles = torch.tensor([[175.0, 170.0]])  # O-H⋯O angles
      
      expanded_hbonds = compute_symmetric_hbonds_batch(
          donor_labels, hydrogen_labels, acceptor_labels,
          donor_idx, hydrogen_idx, acceptor_idx,
          donor_frac, hydrogen_frac, acceptor_frac,
          lengths, angles, in_los,
          symmetry_A, symmetry_T, symmetry_A_inv, symmetry_T_inv,
          cell_matrix, device
      )
      
      # Extract H-bond statistics
      hb_lengths = expanded_hbonds[12]  # D⋯A distances
      hb_angles = expanded_hbonds[13]   # D-H⋯A angles
      hb_mask = expanded_hbonds[-1]     # Valid H-bonds
      
      print(f"Average H-bond length: {hb_lengths[hb_mask].mean():.2f} Å")
      print(f"Average H-bond angle: {hb_angles[hb_mask].mean():.1f}°")

Contact Classification and Validation
-------------------------------------

.. autofunction:: contact_utils.compute_contact_is_hbond

   **Hydrogen Bond Contact Identification**

   Determines which intermolecular contacts correspond to hydrogen bonds by matching contact pairs with hydrogen bond triplets.

   **Algorithm:**

   1. **Contact enumeration** - List all intermolecular atomic contacts
   2. **H-bond enumeration** - List all D-H⋯A hydrogen bond triplets
   3. **Pair matching** - Match contact pairs (D⋯A) with H-bond pairs
   4. **Flag assignment** - Mark contacts that participate in H-bonds

   **Parameters:**

   * **cc_central_idx** (:obj:`torch.LongTensor`, shape (B, C)) - Contact central atom indices
   * **cc_contact_idx** (:obj:`torch.LongTensor`, shape (B, C)) - Contact atom indices
   * **cc_mask** (:obj:`torch.BoolTensor`, shape (B, C)) - Contact validity mask
   * **hb_central_idx** (:obj:`torch.LongTensor`, shape (B, H)) - H-bond donor indices
   * **hb_hydrogen_idx** (:obj:`torch.LongTensor`, shape (B, H)) - H-bond hydrogen indices
   * **hb_contact_idx** (:obj:`torch.LongTensor`, shape (B, H)) - H-bond acceptor indices
   * **hb_mask** (:obj:`torch.BoolTensor`, shape (B, H)) - H-bond validity mask
   * **device** (:obj:`torch.device`) - Computation device

   **Returns:**

   * **torch.BoolTensor**, shape (B, C) - True where contacts participate in hydrogen bonds

   **Implementation Logic:**

   .. code-block:: python

      def identify_hbond_contacts(contact_pairs, hbond_triplets):
          """Match contacts with hydrogen bonds."""
          
          # Extract donor-acceptor pairs from H-bond triplets
          hbond_pairs = hbond_triplets[:, [0, 2]]  # [donor, acceptor]
          
          # Compare each contact pair with H-bond pairs
          contact_is_hbond = torch.zeros(len(contact_pairs), dtype=torch.bool)
          
          for i, contact in enumerate(contact_pairs):
              # Check if contact matches any H-bond pair
              matches = torch.all(contact.unsqueeze(0) == hbond_pairs, dim=1)
              contact_is_hbond[i] = torch.any(matches)
          
          return contact_is_hbond

Fragment-Level Contact Analysis
-------------------------------

.. autofunction:: contact_utils.compute_contact_fragment_indices_batch

   **Map Atomic Contacts to Fragment Interactions**

   Converts atomic-level contact data to fragment-level interactions for higher-level analysis.

   **Parameters:**

   * **contact_central_idx** (:obj:`torch.LongTensor`) - Central atom indices in contacts
   * **contact_atom_idx** (:obj:`torch.LongTensor`) - Contact atom indices
   * **contact_mask** (:obj:`torch.BoolTensor`) - Valid contact indicators
   * **atom_fragment_ids** (:obj:`torch.LongTensor`) - Fragment assignments for atoms
   * **device** (:obj:`torch.device`) - Computation device

   **Returns:**

   * **fragment_central_idx** (:obj:`torch.LongTensor`) - Central fragment indices
   * **fragment_contact_idx** (:obj:`torch.LongTensor`) - Contact fragment indices
   * **fragment_contact_mask** (:obj:`torch.BoolTensor`) - Valid fragment contact mask

   **Applications:**

   * **Packing motif analysis** - Identify common fragment interaction patterns
   * **Polymorph comparison** - Compare fragment contact networks
   * **Crystal engineering** - Design fragment arrangements
   * **Stability analysis** - Correlate contacts with thermodynamic stability

.. autofunction:: contact_utils.compute_contact_atom_to_central_fragment_com_batch

   **Compute Contact Distances to Fragment Centers**

   Calculates distances from contact atoms to the center of mass of the central molecular fragment.

   **Parameters:**

   * **contact_atom_coords** (:obj:`torch.Tensor`) - Contact atom Cartesian coordinates
   * **contact_atom_frac_coords** (:obj:`torch.Tensor`) - Contact atom fractional coordinates
   * **contact_mask** (:obj:`torch.BoolTensor`) - Valid contact indicators
   * **central_fragment_com_coords** (:obj:`torch.Tensor`) - Central fragment COM coordinates
   * **central_fragment_com_frac_coords** (:obj:`torch.Tensor`) - Central fragment COM fractional coordinates
   * **device** (:obj:`torch.device`) - Computation device

   **Returns:**

   * **contact_to_com_distances** (:obj:`torch.Tensor`) - Cartesian distances to COM
   * **contact_to_com_frac_distances** (:obj:`torch.Tensor`) - Fractional distances to COM
   * **contact_to_com_vectors** (:obj:`torch.Tensor`) - Direction vectors to COM

   **Usage in Crystal Analysis:**

   .. code-block:: python

      def analyze_fragment_environment(contact_data, fragment_data):
          """Analyze the local environment around molecular fragments."""
          
          # Compute contact distances to fragment centers
          com_distances = compute_contact_atom_to_central_fragment_com_batch(
              contact_data['contact_coords'],
              contact_data['contact_frac_coords'],
              contact_data['contact_mask'],
              fragment_data['com_coords'],
              fragment_data['com_frac_coords'],
              device
          )
          
          # Analyze distance distribution
          mean_distance = com_distances[0][contact_data['contact_mask']].mean()
          std_distance = com_distances[0][contact_data['contact_mask']].std()
          
          # Identify close approaches
          close_contacts = com_distances[0] < (mean_distance - std_distance)
          
          return {
              'mean_contact_distance': mean_distance,
              'distance_variation': std_distance,
              'close_contact_fraction': close_contacts.float().mean()
          }

Advanced Contact Analysis
-------------------------

**Contact Network Analysis**

.. code-block:: python

   def build_contact_network(fragment_contacts, contact_strengths):
       """Build fragment contact network for graph analysis."""
       import networkx as nx
       
       # Create graph
       G = nx.Graph()
       
       # Add fragment nodes
       fragments = torch.unique(torch.cat([fragment_contacts[:, 0], fragment_contacts[:, 1]]))
       G.add_nodes_from(fragments.tolist())
       
       # Add contact edges with weights
       for i, (frag1, frag2) in enumerate(fragment_contacts):
           if frag1 != frag2:  # No self-contacts
               strength = contact_strengths[i].item()
               G.add_edge(frag1.item(), frag2.item(), weight=strength)
       
       return G

   def analyze_contact_network(contact_network):
       """Analyze topological properties of contact network."""
       import networkx as nx
       
       # Basic network properties
       n_nodes = contact_network.number_of_nodes()
       n_edges = contact_network.number_of_edges()
       density = nx.density(contact_network)
       
       # Centrality measures
       degree_centrality = nx.degree_centrality(contact_network)
       betweenness_centrality = nx.betweenness_centrality(contact_network)
       closeness_centrality = nx.closeness_centrality(contact_network)
       
       # Clustering and modularity
       clustering = nx.average_clustering(contact_network)
       
       return {
           'network_size': n_nodes,
           'contact_count': n_edges,
           'network_density': density,
           'clustering_coefficient': clustering,
           'centrality_measures': {
               'degree': degree_centrality,
               'betweenness': betweenness_centrality,
               'closeness': closeness_centrality
           }
       }

**π-π Interaction Detection**

.. code-block:: python

   def detect_pi_pi_interactions(fragment_data, contact_data, planarity_threshold=0.1):
       """Detect π-π stacking interactions between aromatic fragments."""
       
       # Identify aromatic fragments (highly planar)
       aromatic_mask = fragment_data['planarity_scores'] < planarity_threshold
       aromatic_fragments = torch.where(aromatic_mask)[0]
       
       pi_pi_contacts = []
       
       for contact in contact_data:
           frag1_idx = contact['central_fragment_idx']
           frag2_idx = contact['contact_fragment_idx']
           
           # Check if both fragments are aromatic
           if frag1_idx in aromatic_fragments and frag2_idx in aromatic_fragments:
               
               # Get fragment plane normals
               normal1 = fragment_data['plane_normals'][frag1_idx]
               normal2 = fragment_data['plane_normals'][frag2_idx]
               
               # Check if planes are parallel (π-π stacking)
               angle = torch.acos(torch.abs(torch.dot(normal1, normal2)))
               angle_deg = angle * 180 / torch.pi
               
               # Check distance between fragment centers
               com1 = fragment_data['com_coords'][frag1_idx]
               com2 = fragment_data['com_coords'][frag2_idx]
               distance = torch.norm(com2 - com1)
               
               # π-π criteria: parallel planes (±20°) and appropriate distance (3-4 Å)
               if angle_deg < 20 and 3.0 < distance < 4.5:
                   pi_pi_contacts.append({
                       'fragment1': frag1_idx,
                       'fragment2': frag2_idx,
                       'distance': distance,
                       'angle': angle_deg,
                       'type': 'pi_pi_stacking'
                   })
       
       return pi_pi_contacts

Performance Optimization
------------------------

**Memory Management for Large Contact Sets**

.. code-block:: python

   def optimize_contact_processing(n_structures, max_contacts_per_structure):
       """Optimize memory usage for contact processing."""
       
       # Estimate memory requirements
       bytes_per_contact = 200  # Approximate for all contact data
       total_memory_mb = n_structures * max_contacts_per_structure * bytes_per_contact / 1e6
       
       # Get available GPU memory
       if torch.cuda.is_available():
           available_memory_mb = torch.cuda.get_device_properties(0).total_memory / 1e6
           
           if total_memory_mb > available_memory_mb * 0.8:
               # Reduce batch size
               safe_batch_size = int(available_memory_mb * 0.8 / 
                                   (max_contacts_per_structure * bytes_per_contact / 1e6))
               print(f"Reducing batch size to {safe_batch_size} for memory efficiency")
               return safe_batch_size
       
       return n_structures

**Parallel Contact Analysis**

.. code-block:: python

   def parallel_contact_analysis(contact_datasets, n_workers=4):
       """Process contact analysis in parallel across multiple workers."""
       from multiprocessing import Pool
       
       # Define worker function
       def process_contact_batch(batch_data):
           device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
           return comprehensive_contact_analysis(batch_data, device)
       
       # Split data into chunks
       chunk_size = len(contact_datasets) // n_workers
       chunks = [contact_datasets[i:i+chunk_size] 
                for i in range(0, len(contact_datasets), chunk_size)]
       
       # Process in parallel
       with Pool(n_workers) as pool:
           results = pool.map(process_contact_batch, chunks)
       
       # Combine results
       combined_results = {}
       for result_batch in results:
           for key, value in result_batch.items():
               if key not in combined_results:
                   combined_results[key] = []
               combined_results[key].extend(value)
       
       return combined_results

Error Handling and Validation
------------------------------

**Contact Data Validation**

.. code-block:: python

   def validate_contact_data(contact_coords, contact_mask, symmetry_ops):
       """Validate contact data for processing."""
       
       # Check coordinate validity
       if torch.any(torch.isnan(contact_coords[contact_mask])):
           raise ValueError("NaN coordinates in valid contacts")
       
       if torch.any(torch.isinf(contact_coords[contact_mask])):
           raise ValueError("Infinite coordinates in valid contacts")
       
       # Check symmetry operation validity
       det = torch.det(symmetry_ops)
       if not torch.allclose(torch.abs(det), torch.ones_like(det), atol=1e-3):
           raise ValueError("Invalid symmetry matrices (determinant != ±1)")
       
       # Check distance reasonableness
       distances = torch.norm(contact_coords[:, :, 0] - contact_coords[:, :, 1], dim=-1)
       valid_distances = distances[contact_mask]
       
       if torch.any(valid_distances < 0.5):
           raise ValueError("Unreasonably short contact distances (<0.5 Å)")
       
       if torch.any(valid_distances > 20.0):
           raise ValueError("Unreasonably long contact distances (>20 Å)")

**Debugging and Diagnostics**

.. code-block:: python

   def diagnose_contact_analysis(contact_results):
       """Diagnose contact analysis results for quality control."""
       
       print("Contact Analysis Diagnostics:")
       print(f"  Total contacts processed: {contact_results['n_contacts']}")
       print(f"  Hydrogen bonds identified: {contact_results['n_hbonds']}")
       print(f"  H-bond fraction: {contact_results['n_hbonds']/contact_results['n_contacts']:.3f}")
       
       # Distance distribution
       distances = contact_results['distances']
       print(f"  Distance range: {distances.min():.2f} - {distances.max():.2f} Å")
       print(f"  Mean distance: {distances.mean():.2f} ± {distances.std():.2f} Å")
       
       # Contact type distribution
       if 'contact_types' in contact_results:
           types = contact_results['contact_types']
           unique_types, counts = torch.unique(types, return_counts=True)
           for contact_type, count in zip(unique_types, counts):
               print(f"  {contact_type}: {count} contacts")
       
       # Flag potential issues
       if contact_results['n_hbonds'] / contact_results['n_contacts'] > 0.5:
           print("  Warning: Unusually high H-bond fraction")
       
       if distances.std() > distances.mean():
           print("  Warning: High distance variation")

Integration Examples
--------------------

**Complete Crystal Contact Analysis**

.. code-block:: python

   def comprehensive_crystal_contact_analysis(crystal_data):
       """Perform complete contact analysis on crystal structures."""
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
       # 1. Expand intermolecular contacts
       expanded_contacts = compute_symmetric_contacts_batch(
           crystal_data['central_labels'],
           crystal_data['contact_labels'],
           crystal_data['central_idx'],
           crystal_data['contact_idx'],
           crystal_data['central_frac_coords'],
           crystal_data['contact_frac_coords'],
           crystal_data['lengths'],
           crystal_data['strengths'],
           crystal_data['in_los'],
           crystal_data['symmetry_A'],
           crystal_data['symmetry_T'],
           crystal_data['symmetry_A_inv'],
           crystal_data['symmetry_T_inv'],
           crystal_data['cell_matrices'],
           device
       )
       
       # 2. Expand hydrogen bonds
       expanded_hbonds = compute_symmetric_hbonds_batch(
           crystal_data['hb_donor_labels'],
           crystal_data['hb_hydrogen_labels'],
           crystal_data['hb_acceptor_labels'],
           crystal_data['hb_donor_idx'],
           crystal_data['hb_hydrogen_idx'],
           crystal_data['hb_acceptor_idx'],
           crystal_data['hb_donor_frac_coords'],
           crystal_data['hb_hydrogen_frac_coords'],
           crystal_data['hb_acceptor_frac_coords'],
           crystal_data['hb_lengths'],
           crystal_data['hb_angles'],
           crystal_data['hb_in_los'],
           crystal_data['hb_symmetry_A'],
           crystal_data['hb_symmetry_T'],
           crystal_data['hb_symmetry_A_inv'],
           crystal_data['hb_symmetry_T_inv'],
           crystal_data['cell_matrices'],
           device
       )
       
       # 3. Identify which contacts are hydrogen bonds
       contact_is_hbond = compute_contact_is_hbond(
           expanded_contacts[2],  # central_atom_idx
           expanded_contacts[3],  # contact_atom_idx
           expanded_contacts[-1], # contact_mask
           expanded_hbonds[3],    # hb_central_idx
           expanded_hbonds[4],    # hb_hydrogen_idx
           expanded_hbonds[5],    # hb_contact_idx
           expanded_hbonds[-1],   # hb_mask
           device
       )
       
       # 4. Map to fragment interactions
       fragment_contacts = compute_contact_fragment_indices_batch(
           expanded_contacts[2],  # central_atom_idx
           expanded_contacts[3],  # contact_atom_idx
           expanded_contacts[-1], # contact_mask
           crystal_data['atom_fragment_ids'],
           device
       )
       
       # 5. Compute fragment-level distances
       fragment_com_distances = compute_contact_atom_to_central_fragment_com_batch(
           expanded_contacts[5],  # contact_atom_coords
           expanded_contacts[7],  # contact_atom_frac_coords
           expanded_contacts[-1], # contact_mask
           crystal_data['fragment_com_coords'],
           crystal_data['fragment_com_frac_coords'],
           device
       )
       
       return {
           'expanded_contacts': expanded_contacts,
           'expanded_hbonds': expanded_hbonds,
           'contact_is_hbond': contact_is_hbond,
           'fragment_contacts': fragment_contacts,
           'fragment_distances': fragment_com_distances
       }

**Pharmaceutical Crystal Analysis**

.. code-block:: python

   def analyze_pharmaceutical_contacts(drug_crystal_data):
       """Analyze intermolecular contacts in pharmaceutical crystals."""
       
       # Perform comprehensive contact analysis
       contact_results = comprehensive_crystal_contact_analysis(drug_crystal_data)
       
       # Extract contact statistics
       contact_distances = contact_results['expanded_contacts'][8]  # lengths
       contact_mask = contact_results['expanded_contacts'][-1]
       hbond_mask = contact_results['contact_is_hbond']
       
       # Analyze H-bond patterns
       hbond_distances = contact_distances[hbond_mask & contact_mask]
       non_hbond_distances = contact_distances[(~hbond_mask) & contact_mask]
       
       # Classify contact types
       strong_contacts = contact_distances < 2.5
       moderate_contacts = (contact_distances >= 2.5) & (contact_distances < 3.5)
       weak_contacts = contact_distances >= 3.5
       
       # Drug-like property analysis
       analysis_results = {
           'total_contacts': contact_mask.sum().item(),
           'hydrogen_bonds': hbond_mask.sum().item(),
           'hbond_percentage': (hbond_mask.sum() / contact_mask.sum() * 100).item(),
           'mean_hbond_distance': hbond_distances.mean().item() if len(hbond_distances) > 0 else 0,
           'mean_contact_distance': contact_distances[contact_mask].mean().item(),
           'contact_distribution': {
               'strong': strong_contacts.sum().item(),
               'moderate': moderate_contacts.sum().item(),
               'weak': weak_contacts.sum().item()
           }
       }
       
       return analysis_results

**Polymorph Comparison**

.. code-block:: python

   def compare_polymorph_contacts(polymorph_data_list):
       """Compare contact patterns between different polymorphs."""
       
       polymorph_results = []
       
       for i, polymorph_data in enumerate(polymorph_data_list):
           # Analyze each polymorph
           contact_results = comprehensive_crystal_contact_analysis(polymorph_data)
           
           # Extract key metrics
           contacts = contact_results['expanded_contacts']
           hbonds = contact_results['expanded_hbonds']
           
           contact_count = contacts[-1].sum().item()
           hbond_count = hbonds[-1].sum().item()
           
           # Fragment interaction analysis
           fragment_contacts = contact_results['fragment_contacts']
           unique_fragment_pairs = len(torch.unique(fragment_contacts, dim=0))
           
           polymorph_results.append({
               'polymorph_id': i,
               'contact_count': contact_count,
               'hbond_count': hbond_count,
               'fragment_interactions': unique_fragment_pairs,
               'contact_density': contact_count / polymorph_data['cell_volume'],
               'hbond_density': hbond_count / polymorph_data['cell_volume']
           })
       
       # Compare polymorphs
       print("Polymorph Contact Comparison:")
       for result in polymorph_results:
           print(f"  Polymorph {result['polymorph_id']}:")
           print(f"    Contacts: {result['contact_count']}")
           print(f"    H-bonds: {result['hbond_count']}")
           print(f"    Contact density: {result['contact_density']:.3f} contacts/Ų")
       
       return polymorph_results

**Contact-Based Stability Prediction**

.. code-block:: python

   def predict_stability_from_contacts(crystal_data):
       """Predict relative stability based on contact patterns."""
       
       contact_results = comprehensive_crystal_contact_analysis(crystal_data)
       
       # Extract contact energetics (simplified model)
       contacts = contact_results['expanded_contacts']
       hbonds = contact_results['expanded_hbonds']
       
       contact_distances = contacts[8]  # lengths
       contact_mask = contacts[-1]
       hbond_mask = contact_results['contact_is_hbond']
       
       hbond_distances = contacts[8][hbond_mask & contact_mask]
       hbond_angles = hbonds[13][hbonds[-1]]  # angles
       
       # Simple energy model
       def contact_energy(distance, is_hbond=False):
           if is_hbond:
               # H-bond energy (simplified Morse potential)
               return -5.0 * torch.exp(-2.0 * (distance - 2.8))
           else:
               # van der Waals energy (Lennard-Jones 6-12)
               sigma = 3.5
               epsilon = 0.5
               r_ratio = sigma / distance
               return 4 * epsilon * (r_ratio**12 - r_ratio**6)
       
       # Calculate total interaction energy
       hbond_energies = contact_energy(hbond_distances, is_hbond=True)
       vdw_distances = contact_distances[(~hbond_mask) & contact_mask]
       vdw_energies = contact_energy(vdw_distances, is_hbond=False)
       
       total_energy = hbond_energies.sum() + vdw_energies.sum()
       
       # Normalize by cell volume
       energy_density = total_energy / crystal_data['cell_volume']
       
       stability_metrics = {
           'total_energy': total_energy.item(),
           'energy_density': energy_density.item(),
           'hbond_contribution': hbond_energies.sum().item(),
           'vdw_contribution': vdw_energies.sum().item(),
           'average_hbond_strength': hbond_energies.mean().item() if len(hbond_energies) > 0 else 0,
           'stability_score': -energy_density.item()  # More negative = more stable
       }
       
       return stability_metrics

Cross-References
----------------

**Related CSA Modules:**

* :doc:`../processing/geometry_utils` - Geometric calculations for contact validation
* :doc:`../processing/fragment_utils` - Fragment identification and analysis
* :doc:`../processing/symmetry_utils` - Symmetry operation parsing and application
* :doc:`../processing/cell_utils` - Unit cell transformations for distance calculations
* :doc:`../extraction/structure_post_extraction_processor` - Main contact processing pipeline

**External Dependencies:**

* `PyTorch <https://pytorch.org/>`_ - Tensor operations and GPU acceleration
* `NumPy <https://numpy.org/>`_ - Array operations and mathematical functions
* `NetworkX <https://networkx.org/>`_ - Graph analysis for contact networks (optional)

**Scientific References:**

* Jeffrey, G. A. "An Introduction to Hydrogen Bonding" Oxford University Press (1997)
* Desiraju, G. R. & Steiner, T. "The Weak Hydrogen Bond in Structural Chemistry and Biology" Oxford University Press (1999)
* Groom, C. R. et al. "The Cambridge Structural Database" *Acta Crystallographica B* 72, 171-179 (2016)
* Spek, A. L. "Structure validation in chemical crystallography" *Acta Crystallographica D* 65, 148-155 (2009)
* Gavezzotti, A. "Molecular Aggregation: Structure Analysis and Molecular Simulation of Crystals and Liquids" Oxford University Press (2007)