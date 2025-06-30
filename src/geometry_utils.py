"""
Module: geometry_utils.py

Batch-based utilities for computing geometric descriptors of molecules and fragments using PyTorch. Includes:
  - Best-fit plane / centroid
  - Planarity metrics (RMSD, max deviation, planarity score)
  - Global Steinhardt Q_l order parameters
  - Distances to special crystallographic planes (fractional)
  - Angles between bonds and special planes (fractional)
  - Quaternion computation from rotation matrices

Dependencies
------------
math
numpy
torch
typing
"""
import math
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional

_RAW_PLANE_NORMALS = torch.tensor([
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [1, 1, 0], [1,-1, 0], [1, 0, 1], [1, 0,-1],
    [0, 1, 1], [0, 1,-1],
    [1, 1, 1], [1, 1,-1], [1,-1, 1], [1,-1,-1],
], dtype=torch.float32)
_DENOMINATORS = torch.tensor([4,6], dtype=torch.float32)

def compute_distances_to_crystallographic_planes_frac_batch(
        atom_frac_coords: torch.Tensor,
        atom_mask: torch.BoolTensor,
        device: torch.device
        ) -> torch.Tensor:
    """
    Compute fractional distances of each atom to the 26 special crystallographic planes.

    Parameters
    ----------
    atom_frac_coords : torch.Tensor, shape (B, A, 3)
        Fractional coordinates of atoms, padded to A slots per structure.
    atom_mask : torch.BoolTensor, shape (B, A)
        True for valid atoms, False for padding.
    device : torch.device
        Device on which to perform the computation.

    Returns
    -------
    torch.Tensor, shape (B, A, 26)
        Absolute fractional distances to each plane (13 normals × 2 denominators).
    """
    # Move inputs & constants on to the right device & dtype
    frac = atom_frac_coords.to(device)
    mask = atom_mask.to(device)
    dtype = frac.dtype

    normals = _RAW_PLANE_NORMALS.to(device=device, dtype=dtype)  # (13,3)
    dens    = _DENOMINATORS.to(device=device, dtype=dtype)       # (2,)

    # unit normals: (13,3)
    norms = normals.norm(dim=1)                                  # (13,)
    u_n   = normals / norms.unsqueeze(1)

    # project each atom onto each normal: (B, A, 13)
    proj = frac.matmul(u_n.t()) * mask.unsqueeze(-1).to(dtype)

    # build denominator‐norm products: (13,2)
    D = norms.unsqueeze(1) * dens.unsqueeze(0)                   # (13,2)

    # compute projection × D → (B, A, 13, 2)
    pD = proj.unsqueeze(-1) * D.unsqueeze(0).unsqueeze(0)

    # distance to nearest plane: |pD – round(pD)| / D
    dist = (pD - pD.round()).abs() / D.unsqueeze(0).unsqueeze(0) # (B,A,13,2)
    dist2 = dist.permute(0, 1, 3, 2)                         # → (B, A, 2, 13)
    dist = dist2.reshape(dist2.shape[0], dist2.shape[1], -1)

    # flatten to (B, A, 26)
    return dist.reshape(dist.shape[0], dist.shape[1], -1)

def compute_angles_between_bonds_and_crystallographic_planes_frac_batch(
        atom_frac_coords: torch.Tensor,
        bond_atom1: torch.LongTensor,
        bond_atom2: torch.LongTensor,
        bond_mask: torch.BoolTensor,
        device: torch.device
        ) -> torch.Tensor:
    """
    Compute angles (in degrees) between each bond vector and the 13 crystallographic plane normals.

    Parameters
    ----------
    atom_frac_coords : torch.Tensor, shape (B, A, 3)
        Fractional coordinates of all atoms, padded to A per structure.
    bond_atom1 : torch.LongTensor, shape (B, M)
        Index of the first atom in each bond slot.
    bond_atom2 : torch.LongTensor, shape (B, M)
        Index of the second atom in each bond slot.
    bond_mask : torch.BoolTensor, shape (B, M)
        True for real bonds, False for padding.
    device : torch.device
        Device on which to perform the computation.

    Returns
    -------
    torch.Tensor, shape (B, M, 13)
        Bond–plane angles in degrees; zeros where bond_mask is False.
    """
    # move inputs onto device
    coords = atom_frac_coords.to(device) # (B, A, 3)
    idx1   = bond_atom1.to(device)       # (B, M)
    idx2   = bond_atom2.to(device)       # (B, M)
    mask   = bond_mask.to(device)        # (B, M)
    dtype  = coords.dtype

    # prepare normals
    raw = _RAW_PLANE_NORMALS.to(device=device, dtype=dtype) # (13, 3)
    norms = raw.norm(dim=1)                                 # (13,)
    unit_normals = raw / norms.unsqueeze(1)                 # (13, 3)

    # gather bond endpoints
    B, A, _ = coords.shape
    _, M    = idx1.shape
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, M) # (B, M)
    v1 = coords[batch_idx, idx1]                                         # (B, M, 3)
    v2 = coords[batch_idx, idx2]                                         # (B, M, 3)

    # compute bond vectors, zeroing out padding
    bond_vecs = (v2 - v1) * mask.unsqueeze(-1).to(dtype) # (B, M, 3)
    bond_len  = bond_vecs.norm(dim=2).clamp(min=1e-8)    # (B, M)

    # project onto each normal: dot = |v| cosθ  → (B, M, 13)
    projs = torch.einsum('bmc,fc->bmf', bond_vecs, unit_normals)

    # cosθ = |dot| / ‖v‖  → clamp → arccos → degrees
    cos_theta   = (projs.abs() / bond_len.unsqueeze(-1)).clamp(0.0, 1.0)
    angles_rad  = torch.acos(cos_theta)                  # (B, M, 13)
    angles_deg  = torch.rad2deg(angles_rad)

    return angles_deg  # (B, M, 13)

def compute_atom_vectors_to_point_batch(
        atom_coords: torch.Tensor,        # (B, N, 3)
        atom_frac_coords: torch.Tensor,   # (B, N, 3)
        atom_mask: torch.BoolTensor,      # (B, N)
        com_coords: torch.Tensor,         # (B, 3)
        com_frac_coords: torch.Tensor,    # (B, 3)
        device: torch.device
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute displacement vectors and Euclidean distances from each atom to a reference point.

    Parameters
    ----------
    atom_coords : torch.Tensor, shape (B, N, 3)
        Cartesian coordinates, padded to N per fragment.
    atom_frac_coords : torch.Tensor, shape (B, N, 3)
        Fractional coordinates, padded to N per fragment.
    atom_mask : torch.BoolTensor, shape (B, N)
        True for real atoms, False for padding.
    com_coords : torch.Tensor, shape (B, 3)
        Reference points in Cartesian space.
    com_frac_coords : torch.Tensor, shape (B, 3)
        Reference points in fractional space.
    device : torch.device
        Device for computation.

    Returns
    -------
    dists_cart : torch.Tensor, shape (B, N)
        Euclidean distances in Cartesian space.
    dists_frac : torch.Tensor, shape (B, N)
        Euclidean distances in fractional space.
    vecs_cart : torch.Tensor, shape (B, N, 3)
        Cartesian displacement vectors (atom → point).
    vecs_frac : torch.Tensor, shape (B, N, 3)
        Fractional displacement vectors.
    """
    # 1) Move everything to the target device
    atom_coords      = atom_coords.to(device)
    atom_frac_coords = atom_frac_coords.to(device)
    atom_mask        = atom_mask.to(device)
    com_coords       = com_coords.to(device)
    com_frac_coords  = com_frac_coords.to(device)

    # 2) Expand mask for vector operations
    mask3 = atom_mask.to(atom_coords.dtype).unsqueeze(-1)  # (B, N, 1)

    # 3) Compute displacement vectors and zero out padding
    vecs_cart = (atom_coords - com_coords.unsqueeze(1)) * mask3
    vecs_frac = (atom_frac_coords - com_frac_coords.unsqueeze(1)) * mask3

    # 4) Compute Euclidean distances (norm over last dim)
    dists_cart = vecs_cart.norm(dim=-1)    # (B, N)
    dists_frac = vecs_frac.norm(dim=-1)    # (B, N)

    return { 
        'fragment_atom_to_com_dist':      dists_cart, 
        'fragment_atom_to_com_frac_dist': dists_frac, 
        'fragment_atom_to_com_vec':       vecs_cart, 
        'fragment_atom_to_com_frac_vec':  vecs_frac
        }

def compute_bond_rotatability_batch(
        atom_symbols: List[List[str]],
        bond_atom1_idx: torch.LongTensor,
        bond_atom2_idx: torch.LongTensor,
        bond_is_cyclic: torch.BoolTensor,
        bond_types: List[List[str]],
        bond_is_rotatable_raw: List[List[bool]],
        device: torch.device
        ) -> torch.BoolTensor:
    """
    Determine which bonds are rotatable based on CSD criteria.

    Parameters
    ----------
    atom_symbols : List[List[str]]
        Element symbols per atom for each structure (B × N).
    bond_atom1_idx : torch.LongTensor, shape (B, M)
        Index of the first atom in each bond slot.
    bond_atom2_idx : torch.LongTensor, shape (B, M)
        Index of the second atom in each bond slot.
    bond_is_cyclic : torch.BoolTensor, shape (B, M)
        True if the bond is in a ring.
    bond_types : List[List[str]]
        Bond types (e.g. 'single', 'double', 'triple') per slot.
    bond_is_rotatable_raw : List[List[bool]]
        Original CSD rotatability flags per slot.
    device : torch.device
        Device for the output tensor.

    Returns
    -------
    torch.BoolTensor, shape (B, M)
        True where the bond passes all checks and is rotatable.
    """
    B, M = bond_atom1_idx.shape
    result = torch.zeros((B, M), dtype=torch.bool, device=device)

    idx1_batch = bond_atom1_idx.tolist()
    idx2_batch = bond_atom2_idx.tolist()
    cyclic_batch = bond_is_cyclic.tolist()

    for b in range(B):
        symbols = atom_symbols[b]
        types   = bond_types[b]
        raw     = bond_is_rotatable_raw[b]
        idx1    = idx1_batch[b]
        idx2    = idx2_batch[b]
        cyclic  = cyclic_batch[b]

        nb = len(types)
        # Build adjacency for actual bonds
        adj: Dict[int, List[Tuple[int,int]]] = {i: [] for i in range(len(symbols))}
        for j in range(nb):
            a1, a2 = idx1[j], idx2[j]
            if a1 < 0 or a2 < 0:
                continue
            adj[a1].append((a2, j))
            adj[a2].append((a1, j))

        def is_sp(atom_idx: int) -> bool:
            if symbols[atom_idx] not in ('C', 'N'):
                return False
            non_h = [nbr for nbr, _ in adj[atom_idx] if symbols[nbr] != 'H']
            has_triple = any(types[slot].lower() == 'triple' for _, slot in adj[atom_idx])
            return len(non_h) == 2 and has_triple

        def is_cumulated_double(atom_idx: int) -> bool:
            neigh = adj[atom_idx]
            if len(neigh) != 2:
                return False
            return all(types[slot].lower() == 'double' for _, slot in neigh)

        # Evaluate only actual bonds
        for j in range(nb):
            a1, a2 = idx1[j], idx2[j]
            # 1) Terminal bond?
            non_h1 = [nbr for nbr, _ in adj[a1] if symbols[nbr] != 'H']
            non_h2 = [nbr for nbr, _ in adj[a2] if symbols[nbr] != 'H']
            if len(non_h1) <= 1 or len(non_h2) <= 1:
                continue
            # 2) Ring or non-single?
            if cyclic[j] or types[j].lower() != 'single':
                continue
            # 3) Linear arrangement?
            if is_sp(a1) or is_sp(a2) or (is_cumulated_double(a1) and is_cumulated_double(a2)):
                continue
            # 4) Fallback
            if raw[j]:
                result[b, j] = True

    return result

def compute_bond_angles_batch(
        atom_labels: List[List[str]],
        atom_coords: torch.Tensor,
        atom_mask: torch.BoolTensor,
        bond_atom1_idx: torch.LongTensor,
        bond_atom2_idx: torch.LongTensor,
        bond_mask: torch.BoolTensor,
        device: torch.device
    ) -> Tuple[List[List[str]], torch.Tensor, torch.BoolTensor, torch.LongTensor]:
    """
    Compute all bond angles (i–j–k) in a batch.

    Parameters
    ----------
    atom_labels : List[List[str]]
        Atom labels per structure (B × N).
    atom_coords : torch.Tensor, shape (B, N, 3)
        Cartesian coordinates of atoms.
    atom_mask : torch.BoolTensor, shape (B, N)
        True for real atoms.
    bond_atom1_idx : torch.LongTensor, shape (B, M)
        Index of first atom in each bond slot.
    bond_atom2_idx : torch.LongTensor, shape (B, M)
        Index of second atom in each bond slot.
    bond_mask : torch.BoolTensor, shape (B, M)
        True for real bonds.
    device : torch.device
        Device for computation.

    Returns
    -------
    angle_ids : List[List[str]]
        Per-structure list of “i–j–k” strings.
    angles : torch.Tensor, shape (B, P_max)
        Angle values in degrees (0 where padding).
    mask_ang : torch.BoolTensor, shape (B, P_max)
        True for real angles.
    idx_tensor : torch.LongTensor, shape (B, P_max, 3)
        Atom index triples for each angle.
    """
    # move everything to device
    coords = atom_coords.to(device)                    # (B, N, 3)
    mask_a = atom_mask.to(device)                      # (B, N)
    idx1   = bond_atom1_idx.to(device).long()          # (B, M)
    idx2   = bond_atom2_idx.to(device).long()          # (B, M)
    mask_b = bond_mask.to(device)                      # (B, M)

    B, N, _ = coords.shape
    _, M    = idx1.shape

    # 1) Which bonds are “real”?
    valid = mask_b & mask_a.gather(1, idx1) & mask_a.gather(1, idx2)   # (B, M)

    # 2) Build the M×M m<n mask
    m = torch.arange(M, device=device)
    mm, nn = torch.meshgrid(m, m, indexing='ij')       # each (M, M)
    tril   = mm < nn                                   # (M, M)
    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1) & tril.unsqueeze(0)  # (B, M, M)

    # 3) Detect which bond-pairs share an atom
    idx1m, idx1n = idx1.unsqueeze(2), idx1.unsqueeze(1)  # (B, M, 1), (B, 1, M)
    idx2m, idx2n = idx2.unsqueeze(2), idx2.unsqueeze(1)
    shared_11 = (idx1m == idx1n) & pair_valid
    shared_12 = (idx1m == idx2n) & pair_valid
    shared_21 = (idx2m == idx1n) & pair_valid
    shared_22 = (idx2m == idx2n) & pair_valid

    # combine for convenience
    share1    = shared_11 | shared_12                  # central is idx1m
    share_n1  = shared_11 | shared_21                  # for selecting wing2
    shared_all = shared_11 | shared_12 | shared_21 | shared_22  # (B, M, M)

    # 4) Extract the triples (i, j, k) for each true shared_all[b, m, n]
    #   central atom j:
    central = torch.where(share1, idx1m, idx2m)        # (B, M, M)
    #   wing1 = the “other end” of bond m
    wing1   = torch.where(share1, idx2m, idx1m)
    #   wing2 = the “other end” of bond n
    wing2   = torch.where(share_n1, idx2n, idx1n)

    # 5) Figure out how many angles per structure
    counts = shared_all.view(B, -1).sum(dim=1)         # (B,)
    P_max  = int(counts.max().item())

    # 6) Pack into fixed‐shape tensor + mask
    idx_tensor = torch.full((B, P_max, 3), -1, dtype=torch.long, device=device)
    mask_ang   = torch.zeros((B, P_max),    dtype=torch.bool, device=device)

    for b in range(B):
        m_idx, n_idx = torch.where(shared_all[b])     # each (P_b,)
        P_b = m_idx.size(0)
        if P_b > 0:
            j = central[b, m_idx, n_idx]               # (P_b,)
            i = wing1[b,  m_idx, n_idx]
            k = wing2[b,  m_idx, n_idx]
            idx_tensor[b, :P_b, 0] = i
            idx_tensor[b, :P_b, 1] = j
            idx_tensor[b, :P_b, 2] = k
            mask_ang[b, :P_b]      = True

    # 7) Compute angles in one go
    b_idx = torch.arange(B, device=device).unsqueeze(1)
    ia, ij, ik = idx_tensor.unbind(dim=2)             # each (B, P_max)
    v1  = coords[b_idx, ia] - coords[b_idx, ij]
    v2  = coords[b_idx, ik] - coords[b_idx, ij]
    v1n = v1 / v1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    v2n = v2 / v2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cosθ = (v1n * v2n).sum(dim=-1).clamp(-1.0, 1.0)
    angles = torch.acos(cosθ) * (180.0 / np.pi)
    angles = angles * mask_ang.float()

    # 8) Build string IDs on CPU (cheap: only P_b entries per structure)
    angle_ids: List[List[str]] = []
    for b in range(B):
        ids = []
        L  = mask_ang[b].sum().item()
        for p in range(L):
            i, j, k = idx_tensor[b, p].tolist()
            ids.append(f"{atom_labels[b][i]}-{atom_labels[b][j]}-{atom_labels[b][k]}")
        angle_ids.append(ids)

    return angle_ids, angles, mask_ang, idx_tensor

def compute_torsion_angles_batch(
        atom_labels: List[List[str]],
        atom_coords: torch.Tensor,
        atom_mask: torch.BoolTensor,
        bond_atom1_idx: torch.Tensor,
        bond_atom2_idx: torch.Tensor,
        bond_mask: torch.BoolTensor,
        device: torch.device
        ) -> Tuple[List[List[str]], torch.Tensor, torch.BoolTensor, torch.LongTensor]:
    """
    Compute all torsion angles (i–j–k–l) for each molecule in a batch.

    Parameters
    ----------
    atom_labels : List[List[str]]
        Atom labels per structure (B × N).
    atom_coords : torch.Tensor, shape (B, N, 3)
        Cartesian coordinates.
    atom_mask : torch.BoolTensor, shape (B, N)
        True for real atoms.
    bond_atom1_idx : torch.Tensor, shape (B, M)
        Index of first atom in each bond slot.
    bond_atom2_idx : torch.Tensor, shape (B, M)
        Index of second atom in each bond slot.
    bond_mask : torch.BoolTensor, shape (B, M)
        True for real bonds.
    device : torch.device
        Device for computation.

    Returns
    -------
    torsion_ids : List[List[str]]
        Per-structure list of “i–j–k–l” strings.
    torsions : torch.Tensor, shape (B, T_max)
        Dihedral angles in degrees (0 where padding).
    mask_tor : torch.BoolTensor, shape (B, T_max)
        True for real torsions.
    idx_tensor : torch.LongTensor, shape (B, T_max, 4)
        Atom index quadruplets for each torsion.
    """
    coords = atom_coords.to(device)            # (B, N, 3)
    mask_a = atom_mask.to(device)              # (B, N)
    # ensure bond indices are int64 for gather
    idx1 = bond_atom1_idx.to(device).long()    # (B, M)
    idx2 = bond_atom2_idx.to(device).long()    # (B, M)
    mask_b = bond_mask.to(device)              # (B, M)

    B, N, _ = coords.shape
    # 1) Identify valid bonds
    valid = mask_b & \
            mask_a.gather(1, idx1) & \
            mask_a.gather(1, idx2)                  # (B, M)

    torsion_ids: List[List[str]] = []
    torsion_quads: List[torch.Tensor] = []
    max_T = 0

    for b in range(B):
        idx1_b = idx1[b]                  # (M,)
        idx2_b = idx2[b]
        valid_b = valid[b]

        # 2) Build adjacency matrix from valid bonds
        adj = torch.zeros((N, N), dtype=torch.bool, device=device)
        a = idx1_b[valid_b]
        c = idx2_b[valid_b]
        adj[a, c] = True
        adj[c, a] = True

        # 3) Central bonds j<k
        j_pairs, k_pairs = torch.where(torch.triu(adj, diagonal=1))  # each (P,)
        P = j_pairs.size(0)
        if P == 0:
            torsion_quads.append(torch.zeros((0, 4), dtype=torch.long, device=device))
            torsion_ids.append([])
            continue

        # 4) Neighbor masks for j and k (exclude partner)
        neigh_i = adj[j_pairs].clone()  # (P, N)
        neigh_i[torch.arange(P, device=device), k_pairs] = False
        neigh_l = adj[k_pairs].clone()  # (P, N)
        neigh_l[torch.arange(P, device=device), j_pairs] = False

        # 5) Cartesian product to find all i, j, k, l quads
        mask_il = neigh_i.unsqueeze(2) & neigh_l.unsqueeze(1)  # (P, N, N)
        mask_flat = mask_il.view(P, -1)                         # (P, N*N)
        p_idx, flat_idx = torch.where(mask_flat)               # (T_b,)
        i_idx = flat_idx // N
        l_idx = flat_idx % N
        j_idx = j_pairs[p_idx]
        k_idx = k_pairs[p_idx]

        quads = torch.stack([i_idx, j_idx, k_idx, l_idx], dim=1)  # (T_b, 4)
        torsion_quads.append(quads)

        # 6) Build label strings per quad
        labels_b = atom_labels[b]
        ids_b = [f"{labels_b[i]}-{labels_b[j]}-{labels_b[k]}-{labels_b[l]}"
                 for i, j, k, l in quads.tolist()]
        torsion_ids.append(ids_b)
        max_T = max(max_T, quads.size(0))

    # 7) Pad all batches to (B, max_T)
    idx_tensor = torch.full((B, max_T, 4), -1, dtype=torch.long, device=device)
    mask_tor = torch.zeros((B, max_T), dtype=torch.bool, device=device)
    for b in range(B):
        tb = torsion_quads[b]
        L = tb.size(0)
        if L > 0:
            idx_tensor[b, :L] = tb
            mask_tor[b, :L] = True

    # 8) Batch compute dihedral angles for all quads
    b_idx = torch.arange(B, device=device).unsqueeze(1)
    p1 = coords[b_idx, idx_tensor[:, :, 0]]
    p2 = coords[b_idx, idx_tensor[:, :, 1]]
    p3 = coords[b_idx, idx_tensor[:, :, 2]]
    p4 = coords[b_idx, idx_tensor[:, :, 3]]

    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    b2u = b2 / b2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x = (n1 * n2).sum(dim=-1)
    y = (torch.cross(n2, n1, dim=-1) * b2u).sum(dim=-1)
    torsions = torch.atan2(y, x) * (180.0 / np.pi)
    torsions = torsions * mask_tor.float()

    return torsion_ids, torsions, mask_tor, idx_tensor

def compute_quaternions_from_rotation_matrices(
        R: torch.Tensor,           # (B, 3, 3)
        device: torch.device
        ) -> torch.Tensor:
    """
    Convert rotation matrices into unit quaternions [w, x, y, z] with w ≥ 0.

    Parameters
    ----------
    R : torch.Tensor, shape (B, 3, 3)
        Proper rotation matrices (RᵀR = I, det=+1).
    device : torch.device
        Device for computation.

    Returns
    -------
    torch.Tensor, shape (B, 4)
        Unit quaternions corresponding to each rotation matrix.
    """
    # 1) Move to device & dtype
    R = R.to(device)

    # 2) trace and raw w
    t  = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]        # (B,)
    qw = 0.5 * torch.sqrt(torch.clamp(t + 1.0, min=0.0))   # (B,)

    # 3) safe denominator for x,y,z
    qw_safe = torch.clamp(qw, min=1e-8)

    qx = (R[..., 2, 1] - R[..., 1, 2]) / (4.0 * qw_safe)
    qy = (R[..., 0, 2] - R[..., 2, 0]) / (4.0 * qw_safe)
    qz = (R[..., 1, 0] - R[..., 0, 1]) / (4.0 * qw_safe)

    # 4) stack, normalize, and enforce w ≥ 0
    quats = torch.stack((qw, qx, qy, qz), dim=-1)
    quats = quats / quats.norm(dim=1, keepdim=True)

    neg = quats[:, 0] < 0
    quats[neg] = -quats[neg]

    return quats

def compute_global_steinhardt_order_parameters_batch(
        atom_to_com_vecs: torch.Tensor,        # (B, N, 3)
        atom_mask: torch.BoolTensor,           # (B, N)
        atom_weights: Optional[torch.Tensor],  # (B, N) or None
        device: torch.device,
        l_values: List[int] = [2, 4, 6, 8, 10],
        eps: float = 1e-12
        ) -> torch.Tensor:
    """
    Compute global Steinhardt Q_l order parameters for a batch.

    Parameters
    ----------
    atom_to_com_vecs : torch.Tensor, shape (B, N, 3)
        Positions relative to the center of mass.
    atom_mask : torch.BoolTensor, shape (B, N)
        True for real atoms, False for padding.
    atom_weights : torch.Tensor or None, shape (B, N)
        Per-atom weights, or None for uniform weights.
    device : torch.device
        Device for computation.
    l_values : List[int], optional
        List of ℓ values at which to compute Q_l.
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    torch.Tensor, shape (B, len(l_values))
        Q_l values for each batch entry.
    """
    # Move to device
    atom_to_com_vecs = atom_to_com_vecs.to(device)
    atom_mask        = atom_mask.to(device)
    if atom_weights is not None:
        atom_weights = atom_weights.to(device)

    B, N, _ = atom_to_com_vecs.shape

    # build weight tensor, zeroing out padding
    if atom_weights is not None:
        w = atom_weights * atom_mask.to(atom_weights.dtype)
    else:
        w = atom_mask.to(atom_to_com_vecs.dtype)

    # total weight per molecule
    W_sum = w.sum(dim=1).clamp(min=eps)  # (B,)

    # spherical coords
    x = atom_to_com_vecs[..., 0]
    y = atom_to_com_vecs[..., 1]
    z = atom_to_com_vecs[..., 2]
    r = torch.linalg.norm(atom_to_com_vecs, dim=-1)           # (B, N)
    r_safe = r + eps
    cos_theta = torch.clamp(z / r_safe, -1.0, 1.0)            # (B, N)
    phi = torch.atan2(y, x)                                   # (B, N)

    Qs = torch.zeros((B, len(l_values)), dtype=atom_to_com_vecs.dtype, device=device)

    for idx_l, l in enumerate(l_values):
        sum_m = torch.zeros(B, dtype=atom_to_com_vecs.dtype, device=device)
        for m in range(l + 1):
            # associated Legendre P_l^m(cosθ)
            P_lm = _assoc_legendre(l, m, cos_theta)            # (B, N)
            P_lm = P_lm * atom_mask.to(P_lm.dtype)

            # normalization constant
            norm_lm = math.sqrt((2*l + 1)/(4*math.pi)
                                * math.factorial(l - m)
                                / math.factorial(l + m))
            norm_lm_t = atom_to_com_vecs.new_tensor(norm_lm)   # scalar tensor

            # weighted spherical harmonic component
            P_norm     = P_lm * norm_lm_t                     # (B, N)
            P_weighted = P_norm * w                           # (B, N)

            cos_mphi = torch.cos(m * phi)
            sin_mphi = torch.sin(m * phi)

            real_part = (P_weighted * cos_mphi).sum(dim=1) / W_sum
            imag_part = (P_weighted * sin_mphi).sum(dim=1) / W_sum

            Qlm_sq = real_part.pow(2) + imag_part.pow(2)
            weight = 1.0 if m == 0 else 2.0
            sum_m = sum_m + weight * Qlm_sq

        Qs[:, idx_l] = torch.sqrt((4 * math.pi)/(2*l + 1) * sum_m)

    return Qs

def _assoc_legendre(
        l: int,
        m: int,
        x: torch.Tensor                     # (B, N), dtype float
        ) -> torch.Tensor:
    """
    Compute the associated Legendre polynomial P_l^m(x) via recurrence.

    Parameters
    ----------
    l : int
        Degree of the polynomial.
    m : int
        Order of the polynomial (0 ≤ m ≤ l).
    x : torch.Tensor, shape (B, N)
        Input values in the interval [–1, 1].

    Returns
    -------
    torch.Tensor, shape (B, N)
        Evaluated P_l^m(x) values.
    """
    # base: P_m^m(x)
    if m == 0:
        p_mm = x.new_ones(x.shape)
    else:
        df = 1.0
        for k in range(1, 2*m, 2):
            df *= k
        p_mm = ((-1)**m) * df * (1 - x**2).pow(m / 2)

    if l == m:
        return p_mm

    # next level P_{m+1}^m
    p_m1m = x * (2*m + 1) * p_mm
    if l == m + 1:
        return p_m1m

    # upward recurrence
    p_lm_minus2 = p_mm
    p_lm_minus1 = p_m1m
    for ll in range(m+2, l+1):
        p_lm = ((2*ll - 1) * x * p_lm_minus1
                - (ll + m - 1) * p_lm_minus2) / (ll - m)
        p_lm_minus2, p_lm_minus1 = p_lm_minus1, p_lm

    return p_lm_minus1

def compute_best_fit_plane_batch(
        coords: torch.Tensor,     # (B, N, 3)
        mask: torch.BoolTensor,   # (B, N)
        device: torch.device
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute best-fit plane normals and centroids via SVD.

    Parameters
    ----------
    coords : torch.Tensor, shape (B, N, 3)
        Cartesian coordinates of atoms, padded to N.
    mask : torch.BoolTensor, shape (B, N)
        True for real atoms, False for padding.
    device : torch.device
        Device for computation.

    Returns
    -------
    normals : torch.Tensor, shape (B, 3)
        Unit normals of the best-fit planes (z ≥ 0).
    centroids : torch.Tensor, shape (B, 3)
        Centroid of valid atoms per batch entry.
    """
    # 1) move to device
    coords = coords.to(device)
    mask   = mask.to(device)

    # 2) expand mask and compute centroids
    mask_f   = mask.unsqueeze(-1).to(coords.dtype)       # (B, N, 1)
    counts   = mask_f.sum(dim=1)                         # (B, 1)
    centroids = (coords * mask_f).sum(dim=1) / counts    # (B, 3)

    # 3) center & mask out padding
    centered = (coords - centroids.unsqueeze(1)) * mask_f  # (B, N, 3)

    # 4) batched SVD → Vh: (B, 3, 3)
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)

    # 5) plane normals = last right‐singular vector
    normals = Vh[:, -1, :]  # (B, 3)

    # 6) enforce consistent orientation (z ≥ 0)
    neg_z = normals[:, 2] < 0
    normals[neg_z] = -normals[neg_z]

    return {
        'fragment_plane_centroid': centroids, 
        'fragment_plane_normal':   normals
        }

def compute_planarity_metrics_batch(
        coords: torch.Tensor,     # (B, N, 3)
        mask: torch.BoolTensor,   # (B, N)
        normals: torch.Tensor,    # (B, 3) — unit plane normals, z ≥ 0
        centroids: torch.Tensor,  # (B, 3) — centroids of valid atoms
        device: torch.device,
        decay_width: float = 0.5
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute planarity metrics (RMSD, max deviation, planarity score) for a batch.

    Parameters
    ----------
    coords : torch.Tensor, shape (B, N, 3)
        Cartesian coordinates of atoms, padded to N.
    mask : torch.BoolTensor, shape (B, N)
        True for real atoms.
    normals : torch.Tensor, shape (B, 3)
        Plane normals (unit vectors).
    centroids : torch.Tensor, shape (B, 3)
        Centroids of valid atoms.
    device : torch.device
        Device for computation.
    decay_width : float, optional
        Width parameter for exponential planarity score.

    Returns
    -------
    rmsd : torch.Tensor, shape (B,)
        Root‐mean‐square deviation from the plane.
    max_dev : torch.Tensor, shape (B,)
        Maximum absolute deviation from the plane.
    planarity_score : torch.Tensor, shape (B,)
        Exponential planarity score exp(–rmsd/decay_width).
    """
    # 1) re‐center atoms and mask
    coords = coords.to(device)
    mask   = mask.to(device)
    mask_f = mask.unsqueeze(-1).to(coords.dtype)           # (B, N, 1)
    centered = (coords - centroids.unsqueeze(1)) * mask_f  # (B, N, 3)

    # 2) signed distances to plane
    dists = torch.abs((centered * normals.unsqueeze(1)).sum(dim=-1))  # (B, N)

    # 3) atom counts
    counts = mask.to(dtype=coords.dtype).sum(dim=1)  # (B,)

    # 4) RMSD
    rmsd = torch.sqrt((dists**2 * mask.to(dtype=coords.dtype)).sum(dim=1) / counts)

    # 5) max deviation (ignore padding)
    dists_masked = dists.masked_fill(~mask, float("-inf"))
    max_dev = dists_masked.max(dim=1).values

    # 6) planarity score
    planarity_score = torch.exp(-rmsd / decay_width)

    # 7) handle too‐few‐atoms case (<3)
    invalid = counts < 3
    if invalid.any():
        rmsd[invalid]            = float("nan")
        max_dev[invalid]         = float("nan")
        planarity_score[invalid] = float("nan")

    return { 
        'fragment_planarity_rmsd':    rmsd, 
        'fragment_planarity_max_dev': max_dev, 
        'fragment_planarity_score':   planarity_score
        }

def compute_fragment_pairwise_vectors_and_distances_batch(
        coords: torch.Tensor,
        mask: torch.BoolTensor,
        heavy_mask: torch.BoolTensor,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute pairwise displacement vectors and Euclidean distances between 
    non-hydrogen atoms in each fragment.

    Parameters
    ----------
    coords : torch.Tensor, shape (F, A, 3)
        Cartesian coordinates for each of F fragments, padded to A atoms.
    mask : torch.BoolTensor, shape (F, A)
        True for real atoms slots, False for padding.
    heavy_mask : torch.BoolTensor, shape (F, A)
        True for heavy (non-H) atom slots, False otherwise.
    device : torch.device
        Device on which to perform the computation.

    Returns
    -------
    distances : torch.Tensor, shape (F, A, A)
        Pairwise Euclidean distances. Entry `[f,i,j]` is the distance between
        atom i and j in fragment f if both are heavy & real; zero otherwise.
    vectors : torch.Tensor, shape (F, A, A, 3)
        Pairwise displacement vectors: coords[j]−coords[i] for each heavy-atom
        pair, zero for any non-heavy or padding atom involved.
    atom1_idx : torch.LongTensor, shape (P,)
        The “i” index of each unique heavy-atom pair (i<j), across all fragments.
    atom2_idx : torch.LongTensor, shape (P,)
        The “j” index of each unique heavy-atom pair (i<j), across all fragments.

    Notes
    -----
    The order of entries in `atom1_indices` and `atom2_indices` matches the
    order you’d get by iterating through `torch.nonzero(pair_valid & (j>i))`.
    """
    # move inputs to device
    coords = coords.to(device)      # (F, A, 3)
    mask   = mask.to(device)        # (F, A)
    heavy  = heavy_mask.to(device)  # (F, A)

    # valid heavy‐atom slots
    valid = mask & heavy                      # (F, A)

    # all pairwise diffs & norms
    diffs = coords.unsqueeze(2) - coords.unsqueeze(1)  # (F, A, A, 3)
    dists = diffs.norm(dim=-1)                         # (F, A, A)

    # mask out any pair with H or padding
    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)  # (F, A, A)
    distances = dists * pair_valid.to(dists.dtype)
    vectors   = diffs * pair_valid.unsqueeze(-1).to(diffs.dtype)

    # extract only unique i<j pairs
    upper_valid = torch.triu(pair_valid, diagonal=1)      # (F, A, A)
    frag_idx, i_idx, j_idx = torch.nonzero(upper_valid, as_tuple=True)

    return {
        'fragment_atom_pair_atom1_idx': i_idx, 
        'fragment_atom_pair_atom2_idx': j_idx,
        'fragment_atom_pair_idx':       frag_idx,
        'fragment_atom_pair_dist':      distances, 
        'fragment_atom_pair_vec':       vectors
        } 




