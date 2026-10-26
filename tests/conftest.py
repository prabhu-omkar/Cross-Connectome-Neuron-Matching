import pytest

from flymatch.config import Layout, load_data_config


@pytest.fixture(scope="session")
def layout():
    return Layout(load_data_config())


@pytest.fixture(scope="session")
def built(layout):
    """Skip data-contract tests until `python -m flymatch.run data` has been run."""
    needed = [layout.neurons, layout.edges, layout.edges_np, layout.gt_pairs, layout.gt_types,
              layout.feature_spec, layout.features("mcns"), layout.features("banc")]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        pytest.skip(f"data not built ({missing}); run `python -m flymatch.run data`")
    return layout
