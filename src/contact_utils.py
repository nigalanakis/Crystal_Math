"""
Module: contact_utils.py

Batch utilities for computing and expanding intermolecular contacts and
hydrogen bonds using symmetry operations and mapping to fragment-level
descriptors.

Dependencies
------------
torch
"""
from typing import List, Tuple
import torch

def compute_symmetric_contacts_batch(
        central_atom_label: List[List[str]],
        contact_atom_label: List[List[str]],
        central_atom_idx: torch.Tensor,
        contact_atom_idx: torch.Tensor,
        central_atom_frac_coords: torch.Tensor,
        contact_atom_frac_coords: torch.Tensor,
        lengths: torch.Tensor,
        strengths: torch.Tensor,
        in_los: torch.Tensor,
        symmetry_A: torch.Tensor,
        symmetry_T: torch.Tensor,
        symmetry_A_inv: torch.Tensor,
        symmetry_T_inv: torch.Tensor,
        cell_matrix: torch.Tensor,
        device: torch.device
        ) -> Tuple[
            List[List[str]], List[List[str]],
            torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor,
            torch.Tensor, List[List[str]],
            torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor
            ]:
    """
    Expand intermolecular contacts by applying precomputed symmetry operations.
    
    Parameters
    ----------
    central_atom_label : List[List[str]]
        Original central-atom labels for each structure.
    contact_atom_label : List[List[str]]
        Original contact-atom labels for each structure.
    central_atom_idx : torch.LongTensor, shape (B, C)
        Central-atom indices per contact.
    contact_atom_idx : torch.LongTensor, shape (B, C)
        Contact-atom indices per contact.
    central_atom_frac_coords : torch.Tensor, shape (B, C, 3)
        Fractional coordinates of central atoms.
    contact_atom_frac_coords : torch.Tensor, shape (B, C, 3)
        Fractional coordinates of contact atoms.
    lengths : torch.Tensor, shape (B, C)
        Contact distances.
    strengths : torch.Tensor, shape (B, C)
        Contact strength metrics.
    in_los : torch.Tensor, shape (B, C)
        Line-of-sight contact mask.
    symmetry_A : torch.Tensor, shape (B, C, 3, 3)
        Symmetry rotation matrices.
    symmetry_T : torch.Tensor, shape (B, C, 3)
        Symmetry translation vectors.
    symmetry_A_inv : torch.Tensor, shape (B, C, 3, 3)
        Inverse symmetry rotation matrices.
    symmetry_T_inv : torch.Tensor, shape (B, C, 3)
        Inverse symmetry translation vectors.
    cell_matrix : torch.Tensor, shape (B, 3, 3)
        Real-space cell matrices for each structure.
    device : torch.device
        Device on which to perform computations.
    
    Returns
    -------
    dict
        Dictionary of extended contact parameters:
          - inter_cc_central_atom : List[List[str]]
          - inter_cc_contact_atom : List[List[str]]
          - inter_cc_central_atom_idx : torch.LongTensor, shape (B, 2C)
          - inter_cc_contact_atom_idx : torch.LongTensor, shape (B, 2C)
          - inter_cc_central_atom_coords : torch.Tensor, shape (B, 2C, 3)
          - inter_cc_contact_atom_coords : torch.Tensor, shape (B, 2C, 3)
          - inter_cc_central_atom_frac_coords : torch.Tensor, shape (B, 2C, 3)
          - inter_cc_contact_atom_frac_coords : torch.Tensor, shape (B, 2C, 3)
          - inter_cc_length : torch.Tensor, shape (B, 2C)
          - inter_cc_strength : torch.Tensor, shape (B, 2C)
          - inter_cc_in_los : torch.Tensor, shape (B, 2C)
          - inter_cc_symmetry_A : torch.Tensor, shape (B, 2C, 3, 3)
          - inter_cc_symmetry_T : torch.Tensor, shape (B, 2C, 3)
          - inter_cc_symmetry_A_inv : torch.Tensor, shape (B, 2C, 3, 3)
          - inter_cc_symmetry_T_inv : torch.Tensor, shape (B, 2C, 3)
          - inter_cc_mask : torch.BoolTensor, shape (B, 2C)
    """
    # Move fractional coords to device
    central_atom_frac_coords = central_atom_frac_coords.to(device=device)
    contact_atom_frac_coords = contact_atom_frac_coords.to(device=device)

    # Use fractional coords dtype to cast all other tensors
    dtype = central_atom_frac_coords.dtype

    # Move and cast other tensors
    central_atom_idx = central_atom_idx.to(device=device)
    contact_atom_idx = contact_atom_idx.to(device=device)
    lengths          = lengths.to(device=device, dtype=dtype)
    strengths        = strengths.to(device=device, dtype=dtype)
    in_los           = in_los.to(device=device, dtype=dtype)
    symmetry_A       = symmetry_A.to(device=device, dtype=dtype)
    symmetry_T       = symmetry_T.to(device=device, dtype=dtype)
    symmetry_A_inv   = symmetry_A_inv.to(device=device, dtype=dtype)
    symmetry_T_inv   = symmetry_T_inv.to(device=device, dtype=dtype)
    cell_matrix      = cell_matrix.to(device=device, dtype=dtype)

    B, C, _ = central_atom_frac_coords.shape

    # 1) Compute reversed fractional coords
    central_atom_frac_coords_rev = torch.einsum('bcij,bcj->bci', symmetry_A_inv, contact_atom_frac_coords) + symmetry_T_inv
    contact_atom_frac_coords_rev = torch.einsum('bcij,bcj->bci', symmetry_A_inv, central_atom_frac_coords) + symmetry_T_inv

    # 2) Concatenate original + reversed coords
    central_atom_frac_coords_pre = torch.cat([central_atom_frac_coords, central_atom_frac_coords_rev], dim=1)
    contact_atom_frac_coords_pre = torch.cat([contact_atom_frac_coords, contact_atom_frac_coords_rev], dim=1)

    # 3) Cartesian coords
    central_atom_coords_pre = torch.matmul(central_atom_frac_coords_pre, cell_matrix)
    contact_atom_coords_pre = torch.matmul(contact_atom_frac_coords_pre, cell_matrix)

    # 4) Duplicate metrics and indices
    lengths_pre          = torch.cat([lengths, lengths], dim=1)
    strengths_pre        = torch.cat([strengths, strengths], dim=1)
    in_los_pre           = torch.cat([in_los, in_los], dim=1)
    central_atom_idx_pre = torch.cat([central_atom_idx, contact_atom_idx], dim=1)
    contact_atom_idx_pre = torch.cat([contact_atom_idx, central_atom_idx], dim=1)

    # 5) Prepare extended symmetry matrices
    symmetry_A_ext     = torch.cat([symmetry_A, symmetry_A_inv], dim=1)
    symmetry_T_ext     = torch.cat([symmetry_T, symmetry_T_inv], dim=1)
    symmetry_A_inv_ext = torch.cat([symmetry_A_inv, symmetry_A], dim=1)
    symmetry_T_inv_ext = torch.cat([symmetry_T_inv, symmetry_T], dim=1)

    # 6) Allocate zero-padded outputs
    central_atom_frac_coords_ext = torch.zeros_like(central_atom_frac_coords_pre)
    contact_atom_frac_coords_ext = torch.zeros_like(contact_atom_frac_coords_pre)
    central_atom_coords_ext      = torch.zeros_like(central_atom_coords_pre)
    contact_atom_coords_ext      = torch.zeros_like(contact_atom_coords_pre)
    lengths_ext                  = torch.zeros_like(lengths_pre)
    strengths_ext                = torch.zeros_like(strengths_pre)
    in_los_ext                   = torch.zeros_like(in_los_pre)
    central_atom_idx_ext         = torch.zeros_like(central_atom_idx_pre)
    contact_atom_idx_ext         = torch.zeros_like(contact_atom_idx_pre)

    # 7) Pack valid contacts first
    for b in range(B):
        nC = int((lengths[b] > 0).sum())
        if nC == 0:
            continue
        orig_end = nC
        rev_start = nC
        rev_end = 2 * nC

        # coords
        central_atom_frac_coords_ext[b, :orig_end]         = central_atom_frac_coords_pre[b, :orig_end]
        central_atom_frac_coords_ext[b, rev_start:rev_end] = central_atom_frac_coords_pre[b, C:C + nC]
        contact_atom_frac_coords_ext[b, :orig_end]         = contact_atom_frac_coords_pre[b, :orig_end]
        contact_atom_frac_coords_ext[b, rev_start:rev_end] = contact_atom_frac_coords_pre[b, C:C + nC]

        central_atom_coords_ext[b, :orig_end]         = central_atom_coords_pre[b, :orig_end]
        central_atom_coords_ext[b, rev_start:rev_end] = central_atom_coords_pre[b, C:C + nC]
        contact_atom_coords_ext[b, :orig_end]         = contact_atom_coords_pre[b, :orig_end]
        contact_atom_coords_ext[b, rev_start:rev_end] = contact_atom_coords_pre[b, C:C + nC]

        # metrics
        lengths_ext[b, :orig_end]           = lengths_pre[b, :orig_end]
        lengths_ext[b, rev_start:rev_end]   = lengths_pre[b, C:C + nC]
        strengths_ext[b, :orig_end]         = strengths_pre[b, :orig_end]
        strengths_ext[b, rev_start:rev_end] = strengths_pre[b, C:C + nC]
        in_los_ext[b, :orig_end]            = in_los_pre[b, :orig_end]
        in_los_ext[b, rev_start:rev_end]    = in_los_pre[b, C:C + nC]

        # indices
        central_atom_idx_ext[b, :orig_end]         = central_atom_idx_pre[b, :orig_end]
        central_atom_idx_ext[b, rev_start:rev_end] = central_atom_idx_pre[b, C:C + nC]
        contact_atom_idx_ext[b, :orig_end]         = contact_atom_idx_pre[b, :orig_end]
        contact_atom_idx_ext[b, rev_start:rev_end] = contact_atom_idx_pre[b, C:C + nC]

    # 8) Extend labels 
    central_atom_labels_ext = [orig + rev for orig, rev in zip(central_atom_label, contact_atom_label)]
    contact_atom_labels_ext = [orig + rev for orig, rev in zip(contact_atom_label, central_atom_label)]
    
    # 9) Calculate the contact mask
    inter_cc_mask = lengths_ext > 0

    return {
        "inter_cc_central_atom":             central_atom_labels_ext,
        "inter_cc_contact_atom":             contact_atom_labels_ext,
        "inter_cc_central_atom_idx":         central_atom_idx_ext,
        "inter_cc_contact_atom_idx":         contact_atom_idx_ext,
        "inter_cc_central_atom_coords":      central_atom_coords_ext,
        "inter_cc_contact_atom_coords":      contact_atom_coords_ext,
        "inter_cc_central_atom_frac_coords": central_atom_frac_coords_ext,
        "inter_cc_contact_atom_frac_coords": contact_atom_frac_coords_ext,
        "inter_cc_length":                   lengths_ext,
        "inter_cc_strength":                 strengths_ext,
        "inter_cc_in_los":                   in_los_ext,
        "inter_cc_symmetry_A":               symmetry_A_ext,
        "inter_cc_symmetry_T":               symmetry_T_ext,
        "inter_cc_symmetry_A_inv":           symmetry_A_inv_ext,
        "inter_cc_symmetry_T_inv":           symmetry_T_inv_ext,
        "inter_cc_mask":                     inter_cc_mask
    }

def compute_symmetric_hbonds_batch(
        central_atom_label: List[List[str]],
        hydrogen_atom_label: List[List[str]],
        contact_atom_label: List[List[str]],
        central_atom_idx: torch.Tensor,
        hydrogen_atom_idx: torch.Tensor,
        contact_atom_idx: torch.Tensor,
        central_atom_frac_coords: torch.Tensor,
        hydrogen_atom_frac_coords: torch.Tensor,
        contact_atom_frac_coords: torch.Tensor,
        lengths: torch.Tensor,
        angles: torch.Tensor,
        in_los: torch.Tensor,
        symmetry_A: torch.Tensor,
        symmetry_T: torch.Tensor,
        symmetry_A_inv: torch.Tensor,
        symmetry_T_inv: torch.Tensor,
        cell_matrix: torch.Tensor,
        device: torch.device
        ) -> Tuple[
            List[List[str]], List[List[str]], List[List[str]],
            torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor,
            List[List[str]],
            torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor
            ]:
    """
    Expand intermolecular H-bonds by applying precomputed symmetry operations.
    
    Parameters
    ----------
    central_atom_label : List[List[str]]
        Original donor-atom labels for each structure.
    hydrogen_atom_label : List[List[str]]
        Original hydrogen-atom labels for each structure.
    contact_atom_label : List[List[str]]
        Original acceptor-atom labels for each structure.
    central_atom_idx : torch.LongTensor, shape (B, H)
        Donor-atom indices per H-bond.
    hydrogen_atom_idx : torch.LongTensor, shape (B, H)
        Hydrogen-atom indices per H-bond.
    contact_atom_idx : torch.LongTensor, shape (B, H)
        Acceptor-atom indices per H-bond.
    central_atom_frac_coords : torch.Tensor, shape (B, H, 3)
        Fractional coordinates of donor atoms.
    hydrogen_atom_frac_coords : torch.Tensor, shape (B, H, 3)
        Fractional coordinates of hydrogen atoms.
    contact_atom_frac_coords : torch.Tensor, shape (B, H, 3)
        Fractional coordinates of acceptor atoms.
    lengths : torch.Tensor, shape (B, H)
        H-bond lengths.
    angles : torch.Tensor, shape (B, H)
        H-bond angles.
    in_los : torch.Tensor, shape (B, H)
        Line-of-sight flag per H-bond.
    symmetry_A : torch.Tensor, shape (B, H, 3, 3)
        Symmetry rotation matrices.
    symmetry_T : torch.Tensor, shape (B, H, 3)
        Symmetry translation vectors.
    symmetry_A_inv : torch.Tensor, shape (B, H, 3, 3)
        Inverse symmetry rotation matrices.
    symmetry_T_inv : torch.Tensor, shape (B, H, 3)
        Inverse symmetry translation vectors.
    cell_matrix : torch.Tensor, shape (B, 3, 3)
        Real-space cell matrices for each structure.
    device : torch.device
        Device on which to perform computations.
    
    Returns
    -------
    dict
        Dictionary of extended H-bond parameters:
          - inter_hb_central_atom : List[List[str]]
          - inter_hb_hydrogen_atom : List[List[str]]
          - inter_hb_contact_atom : List[List[str]]
          - inter_hb_central_atom_idx : torch.LongTensor, shape (B, 2H)
          - inter_hb_hydrogen_atom_idx : torch.LongTensor, shape (B, 2H)
          - inter_hb_contact_atom_idx : torch.LongTensor, shape (B, 2H)
          - inter_hb_central_atom_coords : torch.Tensor, shape (B, 2H, 3)
          - inter_hb_hydrogen_atom_coords : torch.Tensor, shape (B, 2H, 3)
          - inter_hb_contact_atom_coords : torch.Tensor, shape (B, 2H, 3)
          - inter_hb_central_atom_frac_coords : torch.Tensor, shape (B, 2H, 3)
          - inter_hb_hydrogen_atom_frac_coords : torch.Tensor, shape (B, 2H, 3)
          - inter_hb_contact_atom_frac_coords : torch.Tensor, shape (B, 2H, 3)
          - inter_hb_length : torch.Tensor, shape (B, 2H)
          - inter_hb_angle : torch.Tensor, shape (B, 2H)
          - inter_hb_in_los : torch.Tensor, shape (B, 2H)
          - inter_hb_symmetry_A : torch.Tensor, shape (B, 2H, 3, 3)
          - inter_hb_symmetry_T : torch.Tensor, shape (B, 2H, 3)
          - inter_hb_symmetry_A_inv : torch.Tensor, shape (B, 2H, 3, 3)
          - inter_hb_symmetry_T_inv : torch.Tensor, shape (B, 2H, 3)
          - inter_hb_mask : torch.BoolTensor, shape (B, 2H)
    """
    # Move fractional coords to device and get dtype
    central_atom_frac_coords  = central_atom_frac_coords.to(device=device)
    hydrogen_atom_frac_coords = hydrogen_atom_frac_coords.to(device=device)
    contact_atom_frac_coords  = contact_atom_frac_coords.to(device=device)
    dtype = central_atom_frac_coords.dtype

    # Move and cast other tensors
    central_atom_idx  = central_atom_idx.to(device=device)
    hydrogen_atom_idx = hydrogen_atom_idx.to(device=device)
    contact_atom_idx  = contact_atom_idx.to(device=device)
    lengths           = lengths.to(device=device, dtype=dtype)
    angles            = angles.to(device=device, dtype=dtype)
    in_los            = in_los.to(device=device, dtype=dtype)
    symmetry_A        = symmetry_A.to(device=device, dtype=dtype)
    symmetry_T        = symmetry_T.to(device=device, dtype=dtype)
    symmetry_A_inv    = symmetry_A_inv.to(device=device, dtype=dtype)
    symmetry_T_inv    = symmetry_T_inv.to(device=device, dtype=dtype)
    cell_matrix       = cell_matrix.to(device=device, dtype=dtype)

    B, C, _ = central_atom_frac_coords.shape

    # 1) Compute reversed fractional coords
    central_atom_frac_coords_rev  = torch.einsum('bcij,bcj->bci', symmetry_A_inv, contact_atom_frac_coords) + symmetry_T_inv
    hydrogen_atom_frac_coords_rev = torch.einsum('bcij,bcj->bci', symmetry_A_inv, hydrogen_atom_frac_coords) + symmetry_T_inv
    contact_atom_frac_coords_rev  = torch.einsum('bcij,bcj->bci', symmetry_A_inv, central_atom_frac_coords) + symmetry_T_inv

    # 2) Concatenate original + reversed coords
    central_atom_frac_coords_pre  = torch.cat([central_atom_frac_coords, central_atom_frac_coords_rev],  dim=1)
    hydrogen_atom_frac_coords_pre = torch.cat([hydrogen_atom_frac_coords, hydrogen_atom_frac_coords_rev], dim=1)
    contact_atom_frac_coords_pre  = torch.cat([contact_atom_frac_coords, contact_atom_frac_coords_rev],  dim=1)

    # 3) Compute Cartesian coords
    central_atom_coords_pre  = torch.matmul(central_atom_frac_coords_pre, cell_matrix)
    hydrogen_atom_coords_pre = torch.matmul(hydrogen_atom_frac_coords_pre, cell_matrix)
    contact_atom_coords_pre  = torch.matmul(contact_atom_frac_coords_pre, cell_matrix)

    # 4) Duplicate metrics
    lengths_pre           = torch.cat([lengths, lengths], dim=1)
    angles_pre            = torch.cat([angles, angles], dim=1)
    in_los_pre            = torch.cat([in_los, in_los], dim=1)
    central_atom_idx_pre  = torch.cat([central_atom_idx, contact_atom_idx], dim=1)
    hydrogen_atom_idx_pre = torch.cat([hydrogen_atom_idx, hydrogen_atom_idx], dim=1)
    contact_atom_idx_pre  = torch.cat([contact_atom_idx, central_atom_idx], dim=1)

    # 5) Prepare extended symmetry matrices
    symmetry_A_ext     = torch.cat([symmetry_A, symmetry_A_inv], dim=1)
    symmetry_T_ext     = torch.cat([symmetry_T, symmetry_T_inv], dim=1)
    symmetry_A_inv_ext = torch.cat([symmetry_A_inv, symmetry_A], dim=1)
    symmetry_T_inv_ext = torch.cat([symmetry_T_inv, symmetry_T], dim=1)

    # 6) Allocate zero-padded outputs
    central_atom_frac_coords_ext  = torch.zeros_like(central_atom_frac_coords_pre)
    hydrogen_atom_frac_coords_ext = torch.zeros_like(hydrogen_atom_frac_coords_pre)
    contact_atom_frac_coords_ext  = torch.zeros_like(contact_atom_frac_coords_pre)
    central_atom_coords_ext       = torch.zeros_like(central_atom_coords_pre)
    hydrogen_atom_coords_ext      = torch.zeros_like(hydrogen_atom_coords_pre)
    contact_atom_coords_ext       = torch.zeros_like(contact_atom_coords_pre)
    lengths_ext                   = torch.zeros_like(lengths_pre)
    angles_ext                    = torch.zeros_like(angles_pre)
    in_los_ext                    = torch.zeros_like(in_los_pre)
    central_atom_idx_ext          = torch.zeros_like(central_atom_idx_pre)
    hydrogen_atom_idx_ext         = torch.zeros_like(hydrogen_atom_idx_pre)
    contact_atom_idx_ext          = torch.zeros_like(contact_atom_idx_pre)

    # 7) Pack valid hbonds first
    for b in range(B):
        nC = int((lengths[b] > 0).sum())
        if nC == 0:
            continue
        orig_end  = nC
        rev_start = nC
        rev_end   = 2 * nC

        # fractional coords
        central_atom_frac_coords_ext[b, :orig_end]          = central_atom_frac_coords_pre[b, :orig_end]
        central_atom_frac_coords_ext[b, rev_start:rev_end]  = central_atom_frac_coords_pre[b, C:C + nC]
        hydrogen_atom_frac_coords_ext[b, :orig_end]         = hydrogen_atom_frac_coords_pre[b, :orig_end]
        hydrogen_atom_frac_coords_ext[b, rev_start:rev_end] = hydrogen_atom_frac_coords_pre[b, C:C + nC]
        contact_atom_frac_coords_ext[b, :orig_end]          = contact_atom_frac_coords_pre[b, :orig_end]
        contact_atom_frac_coords_ext[b, rev_start:rev_end]  = contact_atom_frac_coords_pre[b, C:C + nC]

        # Cartesian coords
        central_atom_coords_ext[b, :orig_end]          = central_atom_coords_pre[b, :orig_end]
        central_atom_coords_ext[b, rev_start:rev_end]  = central_atom_coords_pre[b, C:C + nC]
        hydrogen_atom_coords_ext[b, :orig_end]         = hydrogen_atom_coords_pre[b, :orig_end]
        hydrogen_atom_coords_ext[b, rev_start:rev_end] = hydrogen_atom_coords_pre[b, C:C + nC]
        contact_atom_coords_ext[b, :orig_end]          = contact_atom_coords_pre[b, :orig_end]
        contact_atom_coords_ext[b, rev_start:rev_end]  = contact_atom_coords_pre[b, C:C + nC]

        # metrics
        lengths_ext[b, :orig_end]         = lengths_pre[b, :orig_end]
        lengths_ext[b, rev_start:rev_end] = lengths_pre[b, C:C + nC]
        angles_ext[b, :orig_end]          = angles_pre[b, :orig_end]
        angles_ext[b, rev_start:rev_end]  = angles_pre[b, C:C + nC]
        in_los_ext[b, :orig_end]          = in_los_pre[b, :orig_end]
        in_los_ext[b, rev_start:rev_end]  = in_los_pre[b, C:C + nC]
        
        # indices
        central_atom_idx_ext[b, :orig_end]          = central_atom_idx_pre[b, :orig_end]
        central_atom_idx_ext[b, rev_start:rev_end]  = central_atom_idx_pre[b, C:C + nC]
        hydrogen_atom_idx_ext[b, :orig_end]         = hydrogen_atom_idx_pre[b, :orig_end]
        hydrogen_atom_idx_ext[b, rev_start:rev_end] = hydrogen_atom_idx_pre[b, C:C + nC]
        contact_atom_idx_ext[b, :orig_end]          = contact_atom_idx_pre[b, :orig_end]
        contact_atom_idx_ext[b, rev_start:rev_end]  = contact_atom_idx_pre[b, C:C + nC]

    # 8) Extend label lists
    central_atom_labels_ext  = [orig + rev for orig, rev in zip(central_atom_label, contact_atom_label)]
    hydrogen_atom_labels_ext = [orig + rev for orig, rev in zip(hydrogen_atom_label, hydrogen_atom_label)]
    contact_atom_labels_ext  = [orig + rev for orig, rev in zip(contact_atom_label, central_atom_label)]
    
    # 9) Calculate the contact mask
    inter_hb_mask = lengths_ext > 0

    return {
        "inter_hb_central_atom":              central_atom_labels_ext,
        "inter_hb_hydrogen_atom":             hydrogen_atom_labels_ext,
        "inter_hb_contact_atom":              contact_atom_labels_ext,
        "inter_hb_central_atom_idx":          central_atom_idx_ext,
        "inter_hb_hydrogen_atom_idx":         hydrogen_atom_idx_ext,
        "inter_hb_contact_atom_idx":          contact_atom_idx_ext,
        "inter_hb_central_atom_coords":       central_atom_coords_ext,
        "inter_hb_hydrogen_atom_coords":      hydrogen_atom_coords_ext,
        "inter_hb_contact_atom_coords":       contact_atom_coords_ext,
        "inter_hb_central_atom_frac_coords":  central_atom_frac_coords_ext,
        "inter_hb_hydrogen_atom_frac_coords": hydrogen_atom_frac_coords_ext,
        "inter_hb_contact_atom_frac_coords":  contact_atom_frac_coords_ext,
        "inter_hb_length":                    lengths_ext,
        "inter_hb_angle":                     angles_ext,
        "inter_hb_in_los":                    in_los_ext,
        "inter_hb_symmetry_A":                symmetry_A_ext,
        "inter_hb_symmetry_T":                symmetry_T_ext,
        "inter_hb_symmetry_A_inv":            symmetry_A_inv_ext,
        "inter_hb_symmetry_T_inv":            symmetry_T_inv_ext,
        "inter_hb_mask":                      inter_hb_mask
    }

def compute_contact_is_hbond(
        cc_central_idx: torch.Tensor,
        cc_contact_idx: torch.Tensor,
        cc_mask: torch.Tensor,
        hb_central_idx: torch.Tensor,
        hb_hydrogen_idx: torch.Tensor,
        hb_contact_idx: torch.Tensor,
        hb_mask: torch.Tensor,
        device: torch.device
        ) -> torch.Tensor:
    """
    Flag which contacts correspond to hydrogen bonds.
    
    Parameters
    ----------
    cc_central_idx : torch.LongTensor, shape (B, C)
        Central-atom indices for each intermolecular contact.
    cc_contact_idx : torch.LongTensor, shape (B, C)
        Contact-atom indices for each intermolecular contact.
    cc_mask : torch.BoolTensor, shape (B, C)
        Validity mask for contacts.
    hb_central_idx : torch.LongTensor, shape (B, H)
        Donor-atom indices for each H-bond.
    hb_hydrogen_idx : torch.LongTensor, shape (B, H)
        Hydrogen-atom indices for each H-bond.
    hb_contact_idx : torch.LongTensor, shape (B, H)
        Acceptor-atom indices for each H-bond.
    hb_mask : torch.BoolTensor, shape (B, H)
        Validity mask for H-bonds.
    device : torch.device
        Device on which to perform computations.
    
    Returns
    -------
    torch.BoolTensor, shape (B, C)
        True where each contact participates in any hydrogen-bond triplet.
    """
    # move everything onto the same device
    cc_central_idx = cc_central_idx.to(device)
    cc_contact_idx = cc_contact_idx.to(device)
    cc_mask        = cc_mask.to(device)
    hb_central_idx = hb_central_idx.to(device)
    hb_hydrogen_idx= hb_hydrogen_idx.to(device)
    hb_contact_idx = hb_contact_idx.to(device)
    hb_mask        = hb_mask.to(device)

    # shapes
    B, C_max = cc_central_idx.shape
    _, H_max = hb_central_idx.shape

    # prepare for broadcasting
    cc_c = cc_central_idx.unsqueeze(2)    # (B, C_max, 1)
    cc_p = cc_contact_idx.unsqueeze(2)    # (B, C_max, 1)
    hb_c = hb_central_idx.unsqueeze(1)    # (B, 1, H_max)
    hb_h = hb_hydrogen_idx.unsqueeze(1)   # (B, 1, H_max)
    hb_p = hb_contact_idx.unsqueeze(1)    # (B, 1, H_max)
    hb_m = hb_mask.unsqueeze(1)           # (B, 1, H_max)

    # 1) heavy‐heavy matches
    hh_match = (cc_c == hb_c) & (cc_p == hb_p) & hb_m
    hh_flag  = hh_match.any(dim=2) & cc_mask

    # 2) donor‐to‐H matches
    d2h_match = (cc_c == hb_c) & (cc_p == hb_h) & hb_m
    d2h_flag  = d2h_match.any(dim=2) & cc_mask

    # 3) H‐to‐acceptor matches
    h2a_match = (cc_c == hb_h) & (cc_p == hb_p) & hb_m
    h2a_flag  = h2a_match.any(dim=2) & cc_mask

    # combine all three conditions
    hb_flags = hh_flag | d2h_flag | h2a_flag

    return hb_flags

def compute_contact_fragment_indices_batch(
        central_atom_idx: torch.LongTensor,   # (B, C) atom‐index of the central atom for each contact, -1 if none
        contact_atom_idx: torch.LongTensor,   # (B, C) atom‐index of the other atom in each contact,   -1 if none
        atom_fragment_ids: torch.LongTensor,  # (B, N) fragment ID per atom
        device: torch.device
        ) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Map contact atom indices to fragment IDs.
    
    Parameters
    ----------
    central_atom_idx : torch.LongTensor, shape (B, C)
        Central-atom indices for each contact, or –1 for padding.
    contact_atom_idx : torch.LongTensor, shape (B, C)
        Contact-atom indices for each contact, or –1 for padding.
    atom_fragment_ids : torch.LongTensor, shape (B, N)
        Fragment ID assigned to each atom.
    device : torch.device
        Device on which to perform computations.
    
    Returns
    -------
    dict
        {
          'inter_cc_central_atom_fragment_idx': torch.LongTensor, shape (B, C),
          'inter_cc_contact_atom_fragment_idx': torch.LongTensor, shape (B, C)
        }
    """
    # 1) Move to device and ensure int64
    centr = central_atom_idx.to(device).long()     # now int64
    cont  = contact_atom_idx.to(device).long()
    frag  = atom_fragment_ids.to(device).long()    # ensure int64

    B, N = frag.shape
    _, C = centr.shape

    # 2) clamp into valid range [0, N-1] to avoid out-of-bounds
    zero = torch.zeros((), dtype=torch.long, device=device)
    max_i = torch.full((), N-1, dtype=torch.long, device=device)
    centr_clamped = torch.clamp(centr, zero, max_i)
    cont_clamped  = torch.clamp(cont,  zero, max_i)

    # 3) gather fragment IDs
    central_frag_idx = frag.gather(1, centr_clamped)  # (B, C)
    contact_frag_idx = frag.gather(1, cont_clamped)   # (B, C)

    # 4) restore “–1” where original index was negative
    central_frag_idx = torch.where(centr < 0,
                                   torch.full_like(central_frag_idx, -1),
                                   central_frag_idx)
    contact_frag_idx = torch.where(cont < 0,
                                   torch.full_like(contact_frag_idx, -1),
                                   contact_frag_idx)

    return { 
        'inter_cc_central_atom_fragment_idx': central_frag_idx, 
        'inter_cc_contact_atom_fragment_idx' : contact_frag_idx
        }

def compute_contact_atom_to_central_fragment_com_batch(
        inter_cc_contact_coords: torch.Tensor,       # (B, C_max, 3)
        inter_cc_contact_frac_coords: torch.Tensor,  # (B, C_max, 3)
        central_frag_idx: torch.Tensor,              # (B, Cc), int or long, –1 for padding
        fragment_com_coords: torch.Tensor,           # (F_total, 3)
        fragment_com_frac_coords: torch.Tensor,      # (F_total, 3)
        struct_ids: torch.Tensor,                    # (F_total,), long
        fragment_local_ids: torch.Tensor,            # (F_total,), long
        device: torch.device
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute vectors and distances from contact atoms to central-fragment center of mass.
    
    Parameters
    ----------
    inter_cc_contact_coords : torch.Tensor, shape (B, C, 3)
        Cartesian coordinates of contact atoms.
    inter_cc_contact_frac_coords : torch.Tensor, shape (B, C, 3)
        Fractional coordinates of contact atoms.
    central_frag_idx : torch.LongTensor, shape (B, C)
        Fragment IDs of central atoms for each contact.
    fragment_com_coords : torch.Tensor, shape (F_total, 3)
        Cartesian COM coordinates for all fragments.
    fragment_com_frac_coords : torch.Tensor, shape (F_total, 3)
        Fractional COM coordinates for all fragments.
    struct_ids : torch.Tensor, shape (F_total,)
        Structure IDs corresponding to each fragment.
    fragment_local_ids : torch.Tensor, shape (F_total,)
        Local fragment indices within each structure.
    device : torch.device
        Device on which to perform computations.
    
    Returns
    -------
    dict
        {
          'inter_cc_contact_atom_to_fragment_com_vec': torch.Tensor, shape (B, C, 3),
          'inter_cc_contact_atom_to_fragment_com_frac_vec': torch.Tensor, shape (B, C, 3),
          'inter_cc_contact_atom_to_fragment_com_dist': torch.Tensor, shape (B, C),
          'inter_cc_contact_atom_to_fragment_com_frac_dist': torch.Tensor, shape (B, C)
        }
    """
    # 1) move inputs to device and ensure int64 for indices
    coords_cart = inter_cc_contact_coords.to(device)
    coords_frac = inter_cc_contact_frac_coords.to(device)
    centr_idx   = central_frag_idx.to(device).long()
    com_cart    = fragment_com_coords.to(device).long().float()  # ensure float dtype
    com_frac    = fragment_com_frac_coords.to(device).long().float()
    s_ids       = struct_ids.to(device).long()
    local_ids   = fragment_local_ids.to(device).long()

    B, C_max, _ = coords_cart.shape
    F_total     = com_cart.shape[0]

    # 2) pad central‐fragment indices to C_max columns if needed
    if centr_idx.shape[1] < C_max:
        pad = torch.full((B, C_max - centr_idx.shape[1]), -1, dtype=torch.long, device=device)
        centr_full = torch.cat([centr_idx, pad], dim=1)
    else:
        centr_full = centr_idx

    # 3) build a flattened lookup table of size B * n_frags_max
    #    so that lookup_flat[b * n_frags_max + local_id] = global_row_index
    n_frags_per_struct = torch.bincount(s_ids, minlength=B)            # (B,)
    n_frags_max = int(n_frags_per_struct.max().item())

    lookup_flat = torch.full((B * n_frags_max,), -1, dtype=torch.long, device=device)
    fragment_rows = torch.arange(F_total, device=device, dtype=torch.long)  # (F_total,)
    idx_flat = s_ids * n_frags_max + local_ids                             # (F_total,)
    lookup_flat = lookup_flat.scatter(0, idx_flat, fragment_rows)
    lookup = lookup_flat.view(B, n_frags_max)  # (B, n_frags_max)

    # 4) for each contact, get the global fragment‐COM row index
    #    clamp to [0, n_frags_max-1] to avoid OOB, will mask out invalid entries later
    local_clamped = centr_full.clamp(min=0, max=n_frags_max - 1)  # (B, C_max)
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, C_max)  # (B, C_max)
    com_row_idx = lookup[batch_idx, local_clamped]  # (B, C_max), values in [-1, F_total-1]

    # 5) clamp to valid fragment rows for indexing COM coords
    safe_rows = com_row_idx.clamp(min=0, max=F_total - 1)  # (B, C_max)

    # 6) gather the actual COM coordinates per contact
    point_cart = com_cart[safe_rows]   # (B, C_max, 3)
    point_frac = com_frac[safe_rows]   # (B, C_max, 3)

    # 7) mask for valid contacts, zero out padded ones
    valid_mask = (centr_full >= 0)                     # (B, C_max)
    mask3 = valid_mask.unsqueeze(-1).to(coords_cart.dtype)  # (B, C_max, 1)

    # 8) compute displacement vectors and distances
    vecs_cart = (coords_cart - point_cart) * mask3
    vecs_frac = (coords_frac - point_frac) * mask3

    dists_cart = vecs_cart.norm(dim=-1)  # (B, C_max)
    dists_frac = vecs_frac.norm(dim=-1)  # (B, C_max)

    return {
        'inter_cc_contact_atom_to_fragment_com_dist':      dists_cart, 
        'inter_cc_contact_atom_to_fragment_com_frac_dist': dists_frac, 
        'inter_cc_contact_atom_to_fragment_com_vec':       vecs_cart, 
        'inter_cc_contact_atom_to_fragment_com_frac_vec':  vecs_frac
        }