"""Tests for quorift.config path definitions."""

from pathlib import Path

from quorift.config import ProjectPaths, paths


def test_project_root_structure() -> None:
    """Verify that project_root contains expected repository files and directories."""
    assert paths.project_root.is_dir()
    assert (paths.project_root / "pyproject.toml").is_file()
    assert (paths.project_root / "src").is_dir()


def test_paths_are_relative_to_root() -> None:
    """Verify that all configured paths are subpaths of project_root."""
    root = paths.project_root
    assert paths.data_dir == root / "data"
    assert paths.documents_dir == root / "data" / "documents"
    assert paths.metadata_dir == root / "data" / "metadata"
    assert paths.evaluation_dir == root / "data" / "evaluation"
    assert paths.indexes_dir == root / "data" / "indexes"
    assert paths.reports_dir == root / "reports"
    assert paths.cache_db_path == root / "data" / "cache.sqlite3"


def test_custom_project_root(tmp_path: Path) -> None:
    """Verify that ProjectPaths can be instantiated with a custom root directory."""
    custom_paths = ProjectPaths(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        documents_dir=tmp_path / "data" / "documents",
        metadata_dir=tmp_path / "data" / "metadata",
        evaluation_dir=tmp_path / "data" / "evaluation",
        indexes_dir=tmp_path / "data" / "indexes",
        reports_dir=tmp_path / "reports",
        cache_db_path=tmp_path / "data" / "cache.sqlite3",
    )
    assert custom_paths.project_root == tmp_path
    assert custom_paths.documents_dir == tmp_path / "data" / "documents"
    assert custom_paths.cache_db_path == tmp_path / "data" / "cache.sqlite3"


def test_expected_directory_names() -> None:
    """Verify expected directory basenames for downstream components."""
    assert paths.documents_dir.name == "documents"
    assert paths.metadata_dir.name == "metadata"
    assert paths.evaluation_dir.name == "evaluation"
    assert paths.indexes_dir.name == "indexes"
    assert paths.reports_dir.name == "reports"
    assert paths.cache_db_path.name == "cache.sqlite3"
