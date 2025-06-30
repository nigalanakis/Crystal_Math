"""
Module: fragment_utils.py

Utilities for batch processing of molecular fragments, including rigid‐fragment
identification and computation of per‐fragment properties.

Dependencies
------------
torch
typing
"""
import torch
from typing import List, Tuple, Dict, Union

def identify_rigid_fragments_batch(
        atom_mask: torch.BoolTensor,             # (B, N)
        bond_atom1: torch.LongTensor,            # (B, M)
        bond_atom2: torch.LongTensor,            # (B, M)
        bond_is_rotatable: torch.BoolTensor,     # (B, M)
        device: torch.device
        ) -> torch.LongTensor:
    """
    Identify rigid fragments in a batch via iterative label propagation on GPU.

    Parameters
    ----------
    atom_mask : torch.BoolTensor of shape (B, N)
        True for real atoms, False for padding slots.
    bond_atom1 : torch.LongTensor of shape (B, M)
        First‐atom indices for each bond (–1 for padding).
    bond_atom2 : torch.LongTensor of shape (B, M)
        Second‐atom indices for each bond (–1 for padding).
    bond_is_rotatable : torch.BoolTensor of shape (B, M)
        True if the bond is rotatable; non-rotatable bonds join fragments.
    device : torch.device
        Device to perform computation on (e.g. 'cuda').

    Returns
    -------
    frag_id : torch.LongTensor of shape (B, N)
        Fragment ID for each atom (0..K−1 for real atoms, −1 for padding).
    """
    B, N = atom_mask.shape
    _, M = bond_atom1.shape

    # Move everything to GPU
    atom_mask   = atom_mask.to(device)
    bond_atom1  = bond_atom1.to(device).long()
    bond_atom2  = bond_atom2.to(device).long()
    bond_is_rot = bond_is_rotatable.to(device)

    # 1) initialize each atom’s “label” = its own index, or –1 if padding
    labels = (
        torch.arange(N, device=device)
        .unsqueeze(0).expand(B, N)
        .where(atom_mask, torch.full((B, N), -1, device=device))
        .clone()
    )

    # Precompute which bonds actually tie atoms together
    valid_bond = (bond_atom1 >= 0) & (~bond_is_rot)   # (B, M)

    # For safe gather, clamp negative indices to zero (we’ll mask them out later)
    u_idx = bond_atom1.clamp(min=0)
    v_idx = bond_atom2.clamp(min=0)

    # A “big” label so that invalid bonds never win the min-scatter
    BIG = torch.tensor(N, device=device, dtype=labels.dtype)

    # 2) propagate minima across each non-rotatable bond for up to N iterations
    #    (worst-case chain length = N−1)
    for _ in range(N):
        lu = labels.gather(1, u_idx)  # (B, M)
        lv = labels.gather(1, v_idx)
        mn = torch.min(lu, lv)
        # mask out all rotatable or padding bonds by setting to BIG
        mn = torch.where(valid_bond, mn, BIG)

        # scatter‐reduce the minima back into labels at both endpoints
        labels.scatter_reduce_(1, u_idx, mn, reduce='amin', include_self=True)
        labels.scatter_reduce_(1, v_idx, mn, reduce='amin', include_self=True)

    # 3) remap each unique “root” label to 0..K−1 per batch
    frag_id = torch.full_like(labels, -1)
    for b in range(B):
        lb   = labels[b]                 # (N,)
        mask = atom_mask[b]              # (N,)
        # only consider real-atom labels
        real_labels = lb[mask]
        if real_labels.numel() == 0:
            continue
        # sorted unique roots
        uniq = torch.unique(real_labels)
        # map each atom’s root to its index in uniq
        idx = torch.searchsorted(uniq, lb)
        frag_id[b] = torch.where(mask, idx, torch.tensor(-1, device=device))

    return frag_id

def prepare_fragments_batch(
        atom_fragment_id: torch.LongTensor,
        atom_coords: torch.Tensor,
        atom_frac_coords: torch.Tensor,
        atom_weights: torch.Tensor,
        atom_charges: torch.Tensor,
        atom_symbol_codes: torch.LongTensor,
        code_to_element: List[str],
        code_H: int,
        device: torch.device
        ) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """
    Assemble per‐fragment tensors and compute chemical formulas for a batch.

    Parameters
    ----------
    atom_fragment_id : torch.LongTensor of shape (B, N)
        Fragment index per atom (−1 for padding).
    atom_coords : torch.Tensor of shape (B, N, 3)
        Cartesian coordinates, padded to N atoms.
    atom_frac_coords : torch.Tensor of shape (B, N, 3)
        Fractional coordinates, padded similarly.
    atom_weights : torch.Tensor of shape (B, N)
        Atomic weights (zero for padding).
    atom_charges : torch.Tensor of shape (B, N)
        Partial charges (zero for padding).
    atom_symbol_codes : torch.LongTensor of shape (B, N)
        Integer element codes per atom.
    code_to_element : List[str]
        Mapping from element code to symbol.
    code_H : int
        Integer code corresponding to hydrogen.
    device : torch.device
        Device to perform computation on.

    Returns
    -------
    dict with keys:
      fragment_structure_id : torch.LongTensor of shape (F,)
      fragment_local_id     : torch.LongTensor of shape (F,)
      fragment_n_atoms      : torch.LongTensor of shape (F,)
      fragment_atom_coords  : torch.Tensor of shape (F, max_A, 3)
      fragment_atom_frac_coords : torch.Tensor of shape (F, max_A, 3)
      fragment_atom_weight  : torch.Tensor of shape (F, max_A)
      fragment_atom_charge  : torch.Tensor of shape (F, max_A)
      fragment_atom_mask    : torch.BoolTensor of shape (F, max_A)
      fragment_atom_heavy_mask : torch.BoolTensor of shape (F, max_A)
      fragment_formula      : List[str] of length F
    """
    # Move tensors to device
    atom_fragment_id = atom_fragment_id.to(device)
    atom_coords = atom_coords.to(device)
    atom_frac_coords = atom_frac_coords.to(device)
    atom_weights = atom_weights.to(device)
    atom_charges = atom_charges.to(device)
    atom_symbol_codes = atom_symbol_codes.to(device)

    B, N = atom_fragment_id.shape
    # Determine number of fragments per structure
    n_frags   = (atom_fragment_id.max(dim=1).values + 1).tolist()  # list length B
    n_frags_t = torch.tensor(n_frags, dtype=torch.long, device=device)  # shape (B,)

    # Flatten struct and local fragment IDs
    struct_ids = []
    frag_local_ids = []
    for b in range(B):
        for f in range(n_frags[b]):
            struct_ids.append(b)
            frag_local_ids.append(f)
    F = len(struct_ids)

    struct_ids_t = torch.tensor(struct_ids, dtype=torch.long, device=device)
    frag_local_ids_t = torch.tensor(frag_local_ids, dtype=torch.long, device=device)

    # Build fragment-atom mask (F, N)
    # broadcast struct and local IDs to compare
    # we index per-fragment: mask[i] = atom_fragment_id[struct_ids[i]] == frag_local_ids[i]
    frag_atom_mask_full = atom_fragment_id[struct_ids_t] == frag_local_ids_t.unsqueeze(1)
    frag_n_atoms = frag_atom_mask_full.sum(dim=1)  # (F,)
    max_A = int(frag_n_atoms.max().item())

    # Allocate padded fragment tensors
    frag_coords = torch.zeros((F, max_A, 3), device=device)
    frag_frac = torch.zeros((F, max_A, 3), device=device)
    frag_weights = torch.zeros((F, max_A), device=device)
    frag_charges = torch.zeros((F, max_A), device=device)
    frag_atom_mask = torch.zeros((F, max_A), dtype=torch.bool, device=device)
    frag_heavy_mask = torch.zeros((F, max_A), dtype=torch.bool, device=device)

    fragment_formulas: List[str] = []
    fragment_formulas: List[List[str]] = [[] for _ in range(B)]

    # Fill padded fragment tensors and compute formulas
    for idx in range(F):
        b = struct_ids[idx]
        f = frag_local_ids[idx]
        # atom indices for this fragment
        atom_inds = torch.nonzero(atom_fragment_id[b] == f, as_tuple=False).squeeze(1)
        nA = atom_inds.size(0)
        # Gather properties
        frag_coords[idx, :nA] = atom_coords[b, atom_inds]
        frag_frac[idx, :nA] = atom_frac_coords[b, atom_inds]
        frag_weights[idx, :nA] = atom_weights[b, atom_inds]
        frag_charges[idx, :nA] = atom_charges[b, atom_inds]
        frag_atom_mask[idx, :nA] = True
        # Heavy atom mask via codes
        heavy_src = atom_symbol_codes[b, atom_inds]
        frag_heavy_mask[idx, :nA] = heavy_src != code_H
        # Compute formula counts on CPU
        codes_cpu = heavy_src.cpu().tolist()
        # Count occurrences
        counts: Dict[int,int] = {}
        for c in codes_cpu:
            counts[c] = counts.get(c, 0) + 1
        codes = list(counts.keys())
        codes_sorted = sorted(codes, key=lambda c: code_to_element[c])
        formula = ''.join(
            f"{code_to_element[c]}{counts[c]}"
            for c in codes_sorted
        )
        # fragment_formulas.append(formula)
        fragment_formulas[b].append(formula)

    return {
        'fragment_structure_id':     struct_ids_t,
        'n_fragments':               n_frags_t,
        'fragment_local_id':         frag_local_ids_t,
        'fragment_n_atoms':          frag_n_atoms,
        'fragment_atom_coords':      frag_coords,
        'fragment_atom_frac_coords': frag_frac,
        'fragment_atom_weight':      frag_weights,
        'fragment_atom_charge':      frag_charges,
        'fragment_atom_mask':        frag_atom_mask,
        'fragment_atom_heavy_mask':  frag_heavy_mask,
        'fragment_formula':          fragment_formulas
    }

def compute_center_of_mass_batch(
        atom_coords: torch.Tensor,        # (B, N, 3)
        atom_frac_coords: torch.Tensor,   # (B, N, 3)
        atom_weights: torch.Tensor,       # (B, N)
        atom_mask: torch.BoolTensor,      # (B, N)
        device: torch.device
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Cartesian and fractional centers of mass for each fragment.

    Parameters
    ----------
    atom_coords : torch.Tensor of shape (B, N, 3)
        Cartesian coordinates, padded to N atoms.
    atom_frac_coords : torch.Tensor of shape (B, N, 3)
        Fractional coordinates, padded similarly.
    atom_weights : torch.Tensor of shape (B, N)
        Atomic weights (zero for padding).
    atom_mask : torch.BoolTensor of shape (B, N)
        True for real atoms, False for padding.
    device : torch.device
        Device to perform computation on.

    Returns
    -------
    com_coords : torch.Tensor of shape (B, 3)
        Cartesian center of mass per fragment.
    com_frac_coords : torch.Tensor of shape (B, 3)
        Fractional center of mass per fragment.
    """
    # 1) move everything to device
    atom_coords      = atom_coords.to(device)
    atom_frac_coords = atom_frac_coords.to(device)
    atom_weights     = atom_weights.to(device)
    atom_mask        = atom_mask.to(device)

    # 2) mask out padding atoms (mask → float 0/1)
    w = atom_weights * atom_mask.to(atom_weights.dtype)   # (B, N)

    # 3) total mass per fragment: (B,1)
    total_mass = w.sum(dim=1, keepdim=True)

    # 4) weighted sums → COMs
    #    w.unsqueeze(-1) is (B, N, 1), broadcasts over coords' last dim
    com_coords = (atom_coords * w.unsqueeze(-1)).sum(dim=1) / total_mass
    com_frac_coords = (atom_frac_coords * w.unsqueeze(-1)).sum(dim=1) / total_mass

    return { 
        'fragment_com_coords':      com_coords, 
        'fragment_com_frac_coords': com_frac_coords
        }

def compute_centroid_batch(
        atom_coords: torch.Tensor,        # (B, N, 3)
        atom_frac_coords: torch.Tensor,   # (B, N, 3)
        atom_mask: torch.BoolTensor,      # (B, N)
        device: torch.device
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute geometric centroids in Cartesian and fractional coordinates.

    Parameters
    ----------
    atom_coords : torch.Tensor of shape (B, N, 3)
        Cartesian coordinates, padded to N atoms.
    atom_frac_coords : torch.Tensor of shape (B, N, 3)
        Fractional coordinates, padded similarly.
    atom_mask : torch.BoolTensor of shape (B, N)
        True for real atoms, False for padding.
    device : torch.device
        Device to perform computation on.

    Returns
    -------
    centroid_coords : torch.Tensor of shape (B, 3)
        Cartesian centroids per fragment.
    centroid_frac_coords : torch.Tensor of shape (B, 3)
        Fractional centroids per fragment.
    """
    # 1) move to device
    atom_coords      = atom_coords.to(device)
    atom_frac_coords = atom_frac_coords.to(device)
    atom_mask        = atom_mask.to(device)

    # 2) convert mask to float (1.0 for real atoms, 0.0 for padding)
    m = atom_mask.to(atom_coords.dtype)             # (B, N)

    # 3) count real atoms per fragment: (B,1)
    count = m.sum(dim=1, keepdim=True)

    # avoid division by zero (if a fragment somehow has zero atoms)
    count = torch.clamp(count, min=1.0)

    # 4) sum positions only over real atoms
    sum_cart = (atom_coords * m.unsqueeze(-1)).sum(dim=1)      # (B, 3)
    sum_frac = (atom_frac_coords * m.unsqueeze(-1)).sum(dim=1) # (B, 3)

    # 5) average → centroids
    centroid_coords      = sum_cart / count
    centroid_frac_coords = sum_frac / count

    return {
        'fragment_cen_coords':      centroid_coords, 
        'fragment_cen_frac_coords': centroid_frac_coords
        }

def compute_inertia_tensor_batch(
        atom_coords: torch.Tensor,    # (B, N, 3)
        atom_weights: torch.Tensor,   # (B, N)
        atom_mask: torch.BoolTensor,  # (B, N)
        com_coords: torch.Tensor,     # (B, 3)
        device: torch.device
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute each fragment’s inertia tensor, eigenvalues, and oriented eigenvectors.

    Parameters
    ----------
    atom_coords : torch.Tensor of shape (B, N, 3)
        Cartesian coordinates, padded to N atoms.
    atom_weights : torch.Tensor of shape (B, N)
        Atomic weights, zero for padding.
    atom_mask : torch.BoolTensor of shape (B, N)
        True for real atoms.
    com_coords : torch.Tensor of shape (B, 3)
        Pre‐computed center of mass coordinates.
    device : torch.device
        Device to perform computation on.

    Returns
    -------
    inertia_tensors : torch.Tensor of shape (B, 3, 3)
        Inertia tensor for each fragment.
    eigvals : torch.Tensor of shape (B, 3)
        Eigenvalues (λ₁ ≤ λ₂ ≤ λ₃) per fragment.
    eigvecs : torch.Tensor of shape (B, 3, 3)
        Corresponding right‐handed eigenvectors (columns).
    """
    # Move to device
    atom_coords  = atom_coords.to(device)
    atom_weights = atom_weights.to(device)
    atom_mask    = atom_mask.to(device)
    com_coords   = com_coords.to(device)

    # Expand mask & weights
    mask3 = atom_mask.unsqueeze(-1)               # (B, N, 1)
    w3    = atom_weights.unsqueeze(-1) * mask3    # (B, N, 1)

    # r_i = position relative to COM
    r = (atom_coords - com_coords.unsqueeze(1)) * mask3  # (B, N, 3)

    # r² and outer products
    r2    = (r * r).sum(dim=-1, keepdim=True)     # (B, N, 1)
    outer = r.unsqueeze(-1) * r.unsqueeze(-2)     # (B, N, 3, 3)

    # Identity for broadcasting
    I = torch.eye(3, device=device).view(1, 1, 3, 3)

    # Per‐atom inertia contributions
    #   w * [ (r·r) I₃ − (r⊗r) ]
    terms = w3.unsqueeze(-1) * (r2.unsqueeze(-1) * I - outer)  # (B, N, 3, 3)

    # Sum over atoms → inertia tensor
    inertia_tensors = terms.sum(dim=1)  # (B, 3, 3)

    # Diagonalize
    eigvals, eigvecs = torch.linalg.eigh(inertia_tensors)  # ascending λ's

    # Fix eigenvector signs so each has its max‐abs component ≥ 0
    for i in range(3):
        vec = eigvecs[..., i]                             # (B, 3)
        max_idx = vec.abs().argmax(dim=1, keepdim=True)   # (B, 1)
        sign    = vec.gather(1, max_idx).sign()           # (B, 1)
        sign[sign == 0] = 1.0
        eigvecs[..., i] = vec * sign

    # Enforce right‐handedness
    dets = torch.linalg.det(eigvecs)  # (B,)
    left = dets < 0
    if left.any():
        eigvecs[left, :, 2] *= -1

    return {
        'fragment_inertia_tensors': inertia_tensors, 
        'fragment_inertia_eigvals': eigvals, 
        'fragment_inertia_eigvecs': eigvecs
        }

def compute_quadrupole_tensor_batch(
        atom_coords: torch.Tensor,    # (B, N, 3)
        atom_charges: torch.Tensor,   # (B, N)
        atom_mask: torch.BoolTensor,  # (B, N)
        com_coords: torch.Tensor,     # (B, 3)
        device: torch.device
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute each fragment’s quadrupole tensor, eigenvalues, and eigenvectors.

    Parameters
    ----------
    atom_coords : torch.Tensor of shape (B, N, 3)
        Cartesian coordinates, padded to N atoms.
    atom_charges : torch.Tensor of shape (B, N)
        Atomic charges, zero for padding.
    atom_mask : torch.BoolTensor of shape (B, N)
        True for real atoms.
    com_coords : torch.Tensor of shape (B, 3)
        Pre‐computed center of mass coordinates.
    device : torch.device
        Device to perform computation on.

    Returns
    -------
    quad_tensors : torch.Tensor of shape (B, 3, 3)
        Quadrupole tensor Q per fragment.
    eigvals : torch.Tensor of shape (B, 3)
        Eigenvalues of Q (ascending).
    eigvecs : torch.Tensor of shape (B, 3, 3)
        Right‐handed eigenvectors (columns).
    """
    # Move to device
    atom_coords  = atom_coords.to(device)
    atom_charges = atom_charges.to(device)
    atom_mask    = atom_mask.to(device)
    com_coords   = com_coords.to(device)

    # Masks & charges
    mask3 = atom_mask.unsqueeze(-1)                  # (B, N, 1)
    q4    = atom_charges.unsqueeze(-1).unsqueeze(-1) # (B, N, 1, 1)
    q4    = q4 * mask3.unsqueeze(-1)                 # zero out padding

    # Shift to COM
    r = (atom_coords - com_coords.unsqueeze(1)) * mask3  # (B, N, 3)
    r2    = (r * r).sum(dim=-1, keepdim=True)      # (B, N, 1)
    outer = r.unsqueeze(-1) * r.unsqueeze(-2)      # (B, N, 3, 3)

    I = torch.eye(3, device=device).view(1, 1, 3, 3)

    # Per‐atom quadrupole: q [3 (r⊗r) − |r|² I]
    terms = q4 * (3.0 * outer - r2.unsqueeze(-1) * I)  # (B, N, 3, 3)
    quad_tensors = terms.sum(dim=1)                   # (B, 3, 3)

    # Diagonalize
    eigvals, eigvecs = torch.linalg.eigh(quad_tensors)

    # Fix eigenvector signs (max‐abs component ≥ 0)
    for i in range(3):
        vec = eigvecs[..., i]
        max_idx = vec.abs().argmax(dim=1, keepdim=True)
        sign    = vec.gather(1, max_idx).sign()
        sign[sign == 0] = 1.0
        eigvecs[..., i] = vec * sign

    # Enforce right‐handedness
    dets = torch.linalg.det(eigvecs)
    left = dets < 0
    if left.any():
        eigvecs[left, :, 2] *= -1

    return {
        'fragment_quadrupole_tensors': quad_tensors, 
        'fragment_quadrupole_eigvals': eigvals, 
        'fragment_quadrupole_eigvecs': eigvecs
        }