from pathlib import Path

import pytest

from config import config
from tagolym import main, predict


@pytest.fixture(scope="module")
def artifacts():
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(run_id=run_id)
    return artifacts


@pytest.mark.parametrize(
    "text, tag",
    [
        ("Given a function $f(x) = ax^2 + bx + c$ with integer coefficients.", ("algebra",)),
        ("Given a polynomial $P(x) = ax^2 + bx + c$ with integer coefficients.", ("algebra",)),
    ],
)
def test_inv(text, tag, artifacts):
    """INVariance via verb injection (changes should not affect outputs)."""
    predicted_tag = predict.predict(texts=[text], artifacts=artifacts)[0]["predicted_tags"]
    assert tag == predicted_tag


@pytest.mark.parametrize(
    "text, tag",
    [
        ("Find all real solutions to this equation.", ("algebra",)),
        ("Find all integer solutions to this equation.", ("number theory",)),
    ],
)
def test_dir(text, tag, artifacts):
    """DIRectional expectations (changes with known outputs)."""
    predicted_tag = predict.predict(texts=[text], artifacts=artifacts)[0]["predicted_tags"]
    assert tag == predicted_tag


@pytest.mark.parametrize(
    "text, tag",
    [
        ("Given a cyclic quadrilateral $ABCD$ with $\\angle BAD = \\angle ADC$.", ("geometry",)),
        (
            "How many shortest ways can an ant travel from the lower left to the upper right of a $5 \\times 7$ grid?",
            ("combinatorics",),
        ),
    ],
)
def test_mft(text, tag, artifacts):
    """Minimum Functionality Tests (simple input/output pairs)."""
    predicted_tag = predict.predict(texts=[text], artifacts=artifacts)[0]["predicted_tags"]
    assert tag == predicted_tag
