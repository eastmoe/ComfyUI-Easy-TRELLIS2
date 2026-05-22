"""Sparse 3D grid sampling helper.

The public function mirrors ``flex_gemm.ops.grid_sample.grid_sample_3d`` but
uses sorted sparse-coordinate lookup instead of materializing a dense volume.
"""

import os
from typing import Optional, Tuple

import torch


_DEFAULT_CHUNK_SIZE = 262_144
_TRILINEAR_OFFSETS = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)


def _parse_shape(shape: torch.Size) -> Tuple[Optional[int], int, int, int, int]:
    if len(shape) < 4:
        raise ValueError(f"shape must end with [C, W, H, D], got {shape}")
    channels, width, height, depth = (int(v) for v in shape[-4:])
    batch_size = int(shape[-5]) if len(shape) >= 5 else None
    return batch_size, channels, width, height, depth


def _resolve_chunk_size(chunk_size: Optional[int]) -> int:
    if chunk_size is None:
        chunk_size = int(os.environ.get("TRELLIS2_GRID_SAMPLE_CHUNK_SIZE", _DEFAULT_CHUNK_SIZE))
    return max(1, int(chunk_size))


def _flatten_key_parts(
    batch: int,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    width: int,
    height: int,
    depth: int,
) -> torch.Tensor:
    return (((torch.full_like(x, batch) * width + x) * height + y) * depth + z)


def _flatten_coord_keys(coords: torch.Tensor, width: int, height: int, depth: int) -> torch.Tensor:
    return (((coords[:, 0] * width + coords[:, 1]) * height + coords[:, 2]) * depth + coords[:, 3])


def _lookup(sorted_keys: torch.Tensor, query_keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pos = torch.searchsorted(sorted_keys, query_keys)
    in_range = pos < sorted_keys.shape[0]
    safe_pos = pos.clamp(max=sorted_keys.shape[0] - 1)
    found = in_range & (sorted_keys[safe_pos] == query_keys)
    return safe_pos, found


def _prepare_sparse_index(
    feats: torch.Tensor,
    coords: torch.Tensor,
    batch_limit: int,
    width: int,
    height: int,
    depth: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = coords.to(device=feats.device, dtype=torch.long)
    if coords.dim() != 2 or coords.shape[1] not in (3, 4):
        raise ValueError(f"coords must be [N, 3] or [N, 4], got {coords.shape}")
    if coords.shape[1] == 3:
        coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1)

    valid = (
        (coords[:, 0] >= 0) & (coords[:, 0] < batch_limit) &
        (coords[:, 1] >= 0) & (coords[:, 1] < width) &
        (coords[:, 2] >= 0) & (coords[:, 2] < height) &
        (coords[:, 3] >= 0) & (coords[:, 3] < depth)
    )
    if not bool(valid.all()):
        coords = coords[valid]
        feats = feats[valid]

    keys = _flatten_coord_keys(coords, width, height, depth)
    order = keys.argsort()
    return keys.index_select(0, order), order, feats


def _sample_nearest(
    feats: torch.Tensor,
    sorted_keys: torch.Tensor,
    order: torch.Tensor,
    grid: torch.Tensor,
    width: int,
    height: int,
    depth: int,
    chunk_size: int,
) -> torch.Tensor:
    batch, length = grid.shape[:2]
    channels = feats.shape[1]
    out = feats.new_zeros((batch, length, channels))

    for b in range(batch):
        grid_b = grid[b]
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            pts = grid_b[start:end].float()
            nearest = torch.floor(pts + 0.5).to(torch.long)
            x, y, z = nearest.unbind(dim=-1)
            valid = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (z >= 0) & (z < depth)
            keys = _flatten_key_parts(b, x, y, z, width, height, depth)
            pos, found = _lookup(sorted_keys, keys)
            found = found & valid

            out_chunk = feats.new_zeros((end - start, channels))
            source = order.index_select(0, pos[found])
            out_chunk[found] = feats.index_select(0, source)
            out[b, start:end] = out_chunk

    return out


def _sample_trilinear(
    feats: torch.Tensor,
    sorted_keys: torch.Tensor,
    order: torch.Tensor,
    grid: torch.Tensor,
    width: int,
    height: int,
    depth: int,
    chunk_size: int,
) -> torch.Tensor:
    batch, length = grid.shape[:2]
    channels = feats.shape[1]
    out = feats.new_zeros((batch, length, channels))

    for b in range(batch):
        grid_b = grid[b]
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            pts = grid_b[start:end].float()
            base = torch.floor(pts).to(torch.long)
            frac = pts - base.to(dtype=pts.dtype)
            out_chunk = feats.new_zeros((end - start, channels))

            for ox, oy, oz in _TRILINEAR_OFFSETS:
                x = base[:, 0] + ox
                y = base[:, 1] + oy
                z = base[:, 2] + oz
                wx = frac[:, 0] if ox else (1.0 - frac[:, 0])
                wy = frac[:, 1] if oy else (1.0 - frac[:, 1])
                wz = frac[:, 2] if oz else (1.0 - frac[:, 2])
                weight = wx * wy * wz

                valid = (
                    (weight != 0) &
                    (x >= 0) & (x < width) &
                    (y >= 0) & (y < height) &
                    (z >= 0) & (z < depth)
                )
                keys = _flatten_key_parts(b, x, y, z, width, height, depth)
                pos, found = _lookup(sorted_keys, keys)
                found = found & valid

                source = order.index_select(0, pos[found])
                values = feats.index_select(0, source)
                out_chunk[found] = out_chunk[found] + values * weight[found].to(feats.dtype).unsqueeze(-1)

            out[b, start:end] = out_chunk

    return out


def grid_sample_3d(
    feats: torch.Tensor,
    coords: torch.Tensor,
    shape: torch.Size,
    grid: torch.Tensor,
    mode: str = "trilinear",
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Sample sparse voxel features at absolute grid coordinates.

    Args:
        feats: ``[N, C]`` sparse voxel features.
        coords: ``[N, 4]`` sparse coordinates in ``[batch, x, y, z]`` order.
        shape: dense logical shape ending with ``[C, W, H, D]``.
        grid: ``[B, L, 3]`` query coordinates in voxel index space.
        mode: ``"trilinear"`` or ``"nearest"``.
        chunk_size: maximum query points processed per inner chunk.

    Returns:
        ``[B, L, C]`` sampled features, matching FlexGEMM's API.
    """
    if feats.dim() != 2:
        raise ValueError(f"feats must be [N, C], got {feats.shape}")
    if coords.shape[0] != feats.shape[0]:
        raise ValueError(f"coords and feats must have the same N, got {coords.shape[0]} and {feats.shape[0]}")
    if grid.dim() != 3 or grid.shape[-1] != 3:
        raise ValueError(f"grid must be [B, L, 3], got {grid.shape}")
    if mode not in ("nearest", "trilinear"):
        raise ValueError(f"mode must be 'nearest' or 'trilinear', got {mode!r}")

    shape_batch, shape_channels, width, height, depth = _parse_shape(shape)
    if shape_channels != feats.shape[1]:
        raise ValueError(f"shape channel count {shape_channels} does not match feats {feats.shape[1]}")

    grid = grid.to(device=feats.device)
    batch_limit = shape_batch if shape_batch is not None else grid.shape[0]
    chunk_size = _resolve_chunk_size(chunk_size)

    if feats.shape[0] == 0 or grid.shape[1] == 0:
        return feats.new_zeros((grid.shape[0], grid.shape[1], feats.shape[1]))

    sorted_keys, order, feats = _prepare_sparse_index(
        feats, coords, batch_limit, width, height, depth,
    )
    if sorted_keys.shape[0] == 0:
        return feats.new_zeros((grid.shape[0], grid.shape[1], feats.shape[1]))

    if mode == "nearest":
        return _sample_nearest(feats, sorted_keys, order, grid, width, height, depth, chunk_size)
    return _sample_trilinear(feats, sorted_keys, order, grid, width, height, depth, chunk_size)
