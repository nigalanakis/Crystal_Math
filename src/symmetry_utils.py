"""
Module: symmetry_utils.py

Functions to parse and apply crystallographic symmetry operators, augmenting
parameter dictionaries with rotation matrices and translation vectors.

Dependencies
------------
torch
""" 
import re
from fractions import Fraction
from typing import Dict, Any, List, Tuple

import torch

__all__ = [
    'parse_sym_op',
    'invert_sym_op',
    'add_symmetry_matrices',
    'add_inter_cc_symmetry',
    'add_inter_hb_symmetry',
]


def parse_sym_op(sym: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parse a crystallographic symmetry-operator string into rotation matrix and translation vector.

    Parameters
    ----------
    sym : str
        Symmetry operator string (e.g. 'x+1/2, -y+1/2, z').

    Returns
    -------
    A : torch.Tensor, shape (3, 3)
        Integer rotation matrix.
    t : torch.Tensor, shape (3,)
        Float translation vector.

    Raises
    ------
    ValueError
        If the input string does not contain exactly three comma-separated expressions.
    """
    # remove whitespace, split into 3 axis expressions
    exprs = sym.replace(' ', '').split(',')
    if len(exprs) != 3:
        raise ValueError(f"Invalid symmetry operator: {sym!r}")

    A = torch.zeros((3, 3), dtype=torch.int64)
    t = torch.zeros(3, dtype=torch.float32)
    axis_map = {'x': 0, 'y': 1, 'z': 2}

    for i, expr in enumerate(exprs):
        # 1) find ±x, ±y, ±z terms
        for axis, col in axis_map.items():
            for m in re.finditer(r'([+-]?)(?:1)?' + axis, expr):
                sign = -1 if m.group(1) == '-' else 1
                A[i, col] = sign

        # 2) strip out x/y/z pieces
        const_str = re.sub(r'[+-]?\d*\.?\d*?[xyz]', '', expr)
        if not const_str:
            continue

        # 3) extract numeric tokens
        nums = re.findall(r'[+-]?\d+\.\d+|[+-]?\d+/\d+|[+-]?\d+', const_str)
        for num in nums:
            num = num.rstrip('+-')
            if not re.match(r'[+-]?\d', num):
                continue
            # parse number (fraction or float)
            try:
                if '/' in num:
                    val = float(Fraction(num))
                else:
                    val = float(Fraction(num).limit_denominator())
            except Exception:
                try:
                    val = float(num)
                except Exception:
                    val = 0.0
            t[i] += val

    return A, t


def invert_sym_op(A: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the inverse of a symmetry operator defined by A and t.

    Parameters
    ----------
    A : torch.Tensor, shape (3, 3)
        Rotation matrix.
    t : torch.Tensor, shape (3,)
        Translation vector.

    Returns
    -------
    A_inv : torch.Tensor, shape (3, 3)
        Inverse rotation (transpose of A).
    t_inv : torch.Tensor, shape (3,)
        Inverse translation (-A_inv @ t).

    Raises
    ------
    ValueError
        If A is not shape (3,3) or t is not shape (3,).
    """
    if A.shape != (3, 3) or t.shape != (3,):
        raise ValueError("A must be (3,3) and t must be (3,)")
    A_inv = A.t().contiguous()
    t_inv = -A_inv.to(torch.float32) @ t
    return A_inv, t_inv


def add_symmetry_matrices(
        parameters: Dict[str, Any],
        sym_key: str,
        coords_key: str,
        device: torch.device = None
        ) -> None:
    """
    Add parsed symmetry matrices and their inverses to a parameter dictionary.

    Parameters
    ----------
    parameters : Dict[str, Any]
        Dictionary containing symmetry strings and coordinate tensors.
    sym_key : str
        Key for List[List[str]] of symmetry operator strings in `parameters`.
    coords_key : str
        Key for torch.Tensor of shape (B, N, …) holding coordinates.
    device : torch.device, optional
        Device on which to store the resulting tensors. If None, inferred.

    Modifies
    --------
    parameters : adds the following keys
        '{sym_key}_A'      : torch.Tensor, shape (B, N, 3, 3)
        '{sym_key}_T'      : torch.Tensor, shape (B, N, 3)
        '{sym_key}_A_inv'  : torch.Tensor, shape (B, N, 3, 3)
        '{sym_key}_T_inv'  : torch.Tensor, shape (B, N, 3)

    Raises
    ------
    ValueError
        If input tensor shapes or list lengths are invalid.
    """
    # determine device
    if device is None:
        device = parameters.get('__device__', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # get batch size and max_contacts from coords tensor
    coords = parameters[coords_key]
    if not isinstance(coords, torch.Tensor) or coords.dim() < 2:
        raise ValueError(f"Expected torch.Tensor for '{coords_key}' with dim >=2")
    B, N = coords.shape[0], coords.shape[1]

    # collect symmetry op strings
    sym_ops_batch: List[List[str]] = parameters[sym_key]
    if len(sym_ops_batch) != B:
        raise ValueError(f"Expected {B} lists in '{sym_key}', got {len(sym_ops_batch)}")

    # unique preserving order
    flat = [op for sub in sym_ops_batch for op in sub]
    unique = list(dict.fromkeys(flat))

    # parse unique ops
    A_list = []
    T_list = []
    for op in unique:
        A, t = parse_sym_op(op)
        A_list.append(A)
        T_list.append(t)
    # stack
    A_all = torch.stack(A_list, dim=0).to(device)
    T_all = torch.stack(T_list, dim=0).to(device)

    # compute inverses
    A_inv_all = A_all.permute(0, 2, 1).contiguous()
    # T_inv = -A_inv @ T
    T_inv_all = -(A_inv_all.to(torch.float32) @ T_all.unsqueeze(-1)).squeeze(-1)

    # build index map
    idx_map = {op: idx for idx, op in enumerate(unique)}
    idx = torch.zeros((B, N), dtype=torch.long, device=device)
    for i, row in enumerate(sym_ops_batch):
        for j, op in enumerate(row):
            idx[i, j] = idx_map[op]

    # gather per-contact tensors
    A_per = A_all[idx]        # (B,N,3,3)
    T_per = T_all[idx]        # (B,N,3)
    Ainv_per = A_inv_all[idx] # (B,N,3,3)
    Tinv_per = T_inv_all[idx] # (B,N,3)

    # assign back
    parameters[f'{sym_key}_A'] = A_per
    parameters[f'{sym_key}_T'] = T_per
    parameters[f'{sym_key}_A_inv'] = Ainv_per
    parameters[f'{sym_key}_T_inv'] = Tinv_per


def add_inter_cc_symmetry(parameters: Dict[str, Any], device: torch.device = None) -> None:
    """
    Shortcut to add inter_cc symmetry matrices.

    Parameters
    ----------
    parameters : Dict[str, Any]
        Must contain 'inter_cc_symmetry' and 'inter_cc_central_atom_coords'.
    device : torch.device, optional
        Device for computation.
    """
    add_symmetry_matrices(parameters, 'inter_cc_symmetry', 'inter_cc_central_atom_coords', device)


def add_inter_hb_symmetry(parameters: Dict[str, Any], device: torch.device = None) -> None:
    """
    Shortcut to add inter_hb symmetry matrices.

    Parameters
    ----------
    parameters : Dict[str, Any]
        Must contain 'inter_hb_symmetry' and 'inter_hb_central_atom_coords'.
    device : torch.device, optional
        Device for computation.
    """
    add_symmetry_matrices(parameters, 'inter_hb_symmetry', 'inter_hb_central_atom_coords', device)
