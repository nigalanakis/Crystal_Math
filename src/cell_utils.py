"""
Module: cell_utils.py

Batch utilities for converting unit-cell parameters to normalized/scaled forms
and constructing cell matrices, leveraging PyTorch for GPU acceleration.

Dependencies
------------
torch
"""
import math
import torch
from typing import Optional

__all__ = [
    'compute_scaled_cell',
    'compute_cell_matrix_batch',
    ]


def compute_scaled_cell(
        lengths: torch.Tensor,
        angles: torch.Tensor,
        device: Optional[torch.device] = None
        ) -> torch.Tensor:
    """
    Compute scaled cell parameters for a batch of structures.
    
    Parameters
    ----------
    lengths : torch.Tensor, shape (B, 3)
        Unit-cell lengths [a, b, c] for each structure in the batch.
    angles : torch.Tensor, shape (B, 3)
        Unit-cell angles [α, β, γ] in degrees for each structure in the batch.
    device : torch.device, optional
        Target device for output. If None, uses the device of `lengths`.
    
    Returns
    -------
    torch.Tensor, shape (B, 6)
        Scaled cell feature vectors on the specified device, where each row is
        [1.0, b/a, c/a, α/90, β/90, γ/90].
    """
    if device is None:
        device = lengths.device
    # ensure on correct device
    lengths = lengths.to(device)
    angles = angles.to(device) / 90.0

    # unpack dims: (B,1) each
    a = lengths[:, 0:1]
    b = lengths[:, 1:2]
    c = lengths[:, 2:3]

    # build scaled vector
    ones = torch.ones_like(a)
    scaled = torch.cat([ones, b / a, c / a, angles], dim=1)  # (B,6)
    return scaled


def compute_cell_matrix_batch(
        lengths: torch.Tensor,
        angles: torch.Tensor,
        device: Optional[torch.device] = None
        ) -> torch.Tensor:
    """
    Compute a batch of real-space cell matrices from unit-cell parameters.
    
    Parameters
    ----------
    lengths : torch.Tensor, shape (B, 3)
        Unit-cell lengths [a, b, c] for each structure in the batch.
    angles : torch.Tensor, shape (B, 3)
        Unit-cell angles [α, β, γ] in degrees for each structure in the batch.
    device : torch.device, optional
        Target device for output. If None, uses the device of `lengths`.
    
    Returns
    -------
    torch.Tensor, shape (B, 3, 3)
        Real-space cell matrices on the specified device, where each 3×3 matrix
        has rows:
          [a,      0,                             0]
          [b·cos(γ), b·sin(γ),                   0]
          [c·cos(β), c·(cos(α) – cos(β)·cos(γ))/sin(γ), c_z]
        with c_z = sqrt(c² – M[2,0]² – M[2,1]²).
    """
    if device is None:
        device = lengths.device
    # move to device and convert angles to radians
    lengths = lengths.to(device)
    A = angles.to(device) * (math.pi / 180.0)

    # unpack components
    a, b, c = lengths.unbind(dim=1)                # each (B,)
    cos_a, cos_b, cos_g = torch.cos(A).unbind(dim=1)
    sin_g = torch.sin(A[:, 2])                     # (B,)

    # compute safe c_z
    cz_sq = c**2 - (c * cos_b)**2 - (c * (cos_a - cos_b * cos_g) / sin_g)**2
    cz = torch.sqrt(torch.clamp(cz_sq, min=0.0))   # (B,)

    # build rows
    row0 = torch.stack([a, torch.zeros_like(a), torch.zeros_like(a)], dim=1)
    row1 = torch.stack([b * cos_g, b * sin_g, torch.zeros_like(b)], dim=1)
    row2 = torch.stack([c * cos_b,
                         c * (cos_a - cos_b * cos_g) / sin_g,
                         cz], dim=1)
    # stack into (B,3,3)
    cell = torch.stack([row0, row1, row2], dim=1)
    return cell

