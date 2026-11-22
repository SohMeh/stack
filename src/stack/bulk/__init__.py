"""Bulk RNA-seq support for STACK: convert bulk reference perturbations into single-cell prompts."""

from .expand import (
    ExpansionConfig,
    bulk_fold_changes,
    estimate_nb_theta,
    expand_anchored,
    expand_homogeneous,
    pseudobulk,
    to_cpm,
)

__all__ = [
    "ExpansionConfig",
    "bulk_fold_changes",
    "estimate_nb_theta",
    "expand_anchored",
    "expand_homogeneous",
    "pseudobulk",
    "to_cpm",
]
