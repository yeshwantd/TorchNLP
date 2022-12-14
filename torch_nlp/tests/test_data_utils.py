import pytest, os
import torch_nlp.data.data_utils as data_utils

def test_preprocess_text():
    assert data_utils.preprocess_text("Hello, World! What's up?") == "hello , world ! what s up ? "

def test_preprocess_yelp_csv_file():
    data_utils.preprocess_yelp_csv_file(
        filepath="torch_nlp/tests/test_data/sample_train_data.csv",
        append_with="processed"
    )
    assert os.path.exists("torch_nlp/tests/test_data/sample_train_data_processed.csv")
    os.remove("torch_nlp/tests/test_data/sample_train_data_processed.csv")

def test_count_lines():
    assert data_utils.count_lines(filepath="torch_nlp/tests/test_data/sample_train_data.csv") == 20

def test_split_yelp_data():
    data_utils.split_yelp_data(
        filepath="torch_nlp/tests/test_data/sample_train_data.csv",
        split_ratios=[0.7, 0.3],
        split_filenames=["sample_train_split.csv", "sample_val_split.csv"]
    )
    assert os.path.exists("torch_nlp/tests/test_data/sample_train_split.csv")
    assert os.path.exists("torch_nlp/tests/test_data/sample_val_split.csv")
    assert data_utils.count_lines("torch_nlp/tests/test_data/sample_train_split.csv") == 14
    assert data_utils.count_lines("torch_nlp/tests/test_data/sample_val_split.csv") == 6
    os.remove("torch_nlp/tests/test_data/sample_train_split.csv")
    os.remove("torch_nlp/tests/test_data/sample_val_split.csv")