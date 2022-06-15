import pytest
import numpy as np
import pandas as pd
from tagolym import data

@pytest.fixture(scope="module")
def df():
    data = [
        {"problem": "a0", "tags": ["combinatorics", "number theory"]},
        {"problem": "a1", "tags": ["geometry"]},
        {"problem": "a2", "tags": ["number theory"]},
        {"problem": "a3", "tags": ["geometry", "algebra"]},
        {"problem": "a4", "tags": ["combinatorics"]},
        {"problem": "Given", "tags": ["algebra", "number theory"]},
    ]
    df = pd.DataFrame(data * 10)
    return df

@pytest.fixture(scope="module")
def tags_true():
    data = np.array([
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 1],
        ])
    tags_true = np.tile(data, (10, 1))    
    return tags_true

@pytest.mark.parametrize(
    "text, nocommand, stem, cleaned_text",
    [
        ("Given a triangle $ABC$ where $$\\angle ABC = 90 \\degree$$. Prove \[AB^2 + BC^2 = CA^2\].", False, False, "given triangle angle degree prove"),
        ("Given a triangle $ABC$ where $$\\angle ABC = 90 \\degree$$. Prove \[AB^2 + BC^2 = CA^2\].", True, True, "triangl angl degre"),
    ],
)
def test_preprocess_problem(text, nocommand, stem, cleaned_text):
    assert (
        data.preprocess_problem(
            text,
            nocommand=nocommand,
            stem=stem,
        )
        == cleaned_text
    )

def test_preprocess(df):
    df_preprocessed = data.preprocess(df.copy(), nocommand=True, stem=True)
    assert df_preprocessed.shape == (50, 3)
    assert df_preprocessed.columns.tolist() == ["problem", "tags", "token"]

def test_binarize(df, tags_true):
    tags, mlb = data.binarize(df.copy()["tags"])
    assert np.allclose(tags, tags_true)
    assert mlb.classes_.tolist() == ["algebra", "combinatorics", "geometry", "number theory"]

def test_split_data(df, tags_true):
    X_train, X_val, X_test, y_train, y_val, y_test = data.split_data(df.copy()[["problem"]], tags_true, train_size=0.6)
    assert X_train.shape == (36, 1)
    assert X_val.shape == (12, 1)
    assert X_test.shape == (12, 1)
    assert y_train.shape == (36, 4)
    assert y_val.shape == (12, 4)
    assert y_test.shape == (12, 4)