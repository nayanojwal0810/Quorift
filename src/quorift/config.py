"""Shared project and data paths for Quorift."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Project directory and file paths resolved relative to repository root."""

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )
    data_dir: Path = field(init=False)
    documents_dir: Path = field(init=False)
    metadata_dir: Path = field(init=False)
    evaluation_dir: Path = field(init=False)
    indexes_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    cache_db_path: Path = field(init=False)

    def __post_init__(self) -> None:
        data = self.project_root / "data"
        object.__setattr__(self, "data_dir", data)
        object.__setattr__(self, "documents_dir", data / "documents")
        object.__setattr__(self, "metadata_dir", data / "metadata")
        object.__setattr__(self, "evaluation_dir", data / "evaluation")
        object.__setattr__(self, "indexes_dir", data / "indexes")
        object.__setattr__(self, "reports_dir", self.project_root / "reports")
        object.__setattr__(self, "cache_db_path", data / "cache.sqlite3")


paths = ProjectPaths()
