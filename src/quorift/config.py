"""Shared project and data paths for Quorift."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Project directory and file paths resolved relative to repository root."""

    project_root: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = project_root / "data"
    documents_dir: Path = data_dir / "documents"
    metadata_dir: Path = data_dir / "metadata"
    evaluation_dir: Path = data_dir / "evaluation"
    indexes_dir: Path = data_dir / "indexes"
    reports_dir: Path = project_root / "reports"
    cache_db_path: Path = data_dir / "cache.sqlite3"


paths = ProjectPaths()
