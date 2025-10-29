# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import torch
import math
from typing import Callable, Tuple

def do_nothing(x, mode="mean"):
    return x

def knn_augment(
        tokens: torch.Tensor,
        k: int = 1,
        distance_metric: str = "euclidean"
) -> torch.Tensor:
    """
    fixed-radius k-nearest neighbor strategy ：
    - Calculate the neighbourhood with a radius of 1 for each token (fixed K value).
    - Use a binary mask to avoid recalculating tokens that have already been fused within the window.
    - Euclidean distance is used by default (can be switched via distance_metric).
    """
    n, t, c = tokens.shape
    augmented_tokens = tokens.clone()

    # Calculate the distance matrix between all tokens
    if distance_metric == "euclidean":
        dists = torch.cdist(tokens, tokens, p=2)  # L2 distance
    elif distance_metric == "manhattan":
        dists = torch.cdist(tokens, tokens, p=1)  # L1 distance
    elif distance_metric == "chebyshev":
        dists = torch.cdist(tokens, tokens, p=float('inf'))  # Chebyshev distance
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    # Select the nearest k neighbours for each token (excluding itself)
    mask = torch.eye(t, device=tokens.device).bool()
    dists.masked_fill_(mask, float('inf'))
    _, knn_indices = torch.topk(dists, k=k, dim=-1, largest=False)

    # Fuse each token with its KNN neighbours (equal weights)
    for i in range(t):
        neighbors = tokens[:, knn_indices[:, i], :]
        augmented_tokens[:, i, :] = 0.7 * tokens[:, i, :] + 0.3 * neighbors.mean(dim=1)

    return augmented_tokens


def contextual_window_augment(
        tokens: torch.Tensor,
        window_size: int = 4
) -> torch.Tensor:
    """
    adaptive contextual window mechanism ：
    - Keep the first and last tokens directly.
    - The middle section is grouped according to window_size, and after calculating the mean within the window, it is fused with weighting.
    """
    n, t, c = tokens.shape
    augmented_tokens = tokens.clone()

    if t <= 1:
        return tokens

    # Process the first token separately
    if t > 1:
        # The window for the second token is [0, 1, 2] (if it exists)
        if t >= 3:
            window = tokens[:, 0:3, :]
            augmented_tokens[:, 1, :] = 0.7 * tokens[:, 1, :] + 0.3 * window.mean(dim=1)

    # Group the middle section by window
    for i in range(2, t - 2, window_size):
        if i + window_size <= t:
            window = tokens[:, i:i + window_size, :]
            mean = window.mean(dim=1, keepdim=True)
            augmented_tokens[:, i:i + window_size, :] = 0.7 * window + 0.3 * mean.expand(-1, window_size, -1)

    # Handle the last token separately
    if t >= 2:
        # The window for the penultimate token is [t-3, t-2, t-1] (if it exists)
        if t >= 3:
            window = tokens[:, -3:, :]
            augmented_tokens[:, -2, :] = 0.7 * tokens[:, -2, :] + 0.3 * window.mean(dim=1)

    return augmented_tokens


def bipartite_soft_matching_xincheng(
        metric: torch.Tensor,
        r: int,
        class_token: bool = False,
        distill_token: bool = False,
        window_size: int = 4,
        knn_k: int = 1
) -> Tuple[Callable, Callable]:

    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return lambda x: x, lambda x: x

    with torch.no_grad():
        # If you only want to use one of the methods to enhance your image information, please comment out the others in the corresponding method.

        # Step 1: adaptive contextual window mechanism
        metric = contextual_window_augment(metric, window_size=window_size)

        # Step 2: fixed-radius k-nearest neighbor strategy
        metric = knn_augment(metric, k=knn_k, distance_metric="euclidean")

        # Step 3: Group and calculate similarity (with L2 distance penalty)
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]

        # Similarity score (cosine similarity)
        sim_scores = a @ b.transpose(-1, -2)

        # L2 distance penalty term (normalized to [0, 1])
        dist_penalty = torch.cdist(a, b, p=2)
        dist_penalty = dist_penalty / dist_penalty.max()

        # Overall score = similarity - λ * distance (λ is the balance factor)
        combined_scores = sim_scores - 0.8 * dist_penalty

        if class_token:
            combined_scores[..., 0, :] = -math.inf
        if distill_token:
            combined_scores[..., :, 0] = -math.inf

        node_max, node_idx = combined_scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:

    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge

def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size

def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source
