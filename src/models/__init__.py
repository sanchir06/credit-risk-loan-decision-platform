"""
src/models/__init__.py
======================

Production model artifact management package.

Provides centralized access to artifact paths, serialization utilities,
and artifact validation for the Phase 5 inference layer.

Exported:
    ModelArtifacts  — Path definitions and load/save utilities
    ArtifactError   — Raised when artifacts are missing or inconsistent
"""

from src.models.model_artifacts import ModelArtifacts, ArtifactError

__all__ = ["ModelArtifacts", "ArtifactError"]
