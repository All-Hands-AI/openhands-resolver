import os
import pandas as pd
import pytest
from github_resolver.utils import prepare_dataset

@pytest.fixture
def sample_dataset():
    return pd.DataFrame({
        'instance_id': ['1', '2', '3', '4', '5'],
        'data': ['a', 'b', 'c', 'd', 'e']
    })

def test_prepare_dataset_new_file(sample_dataset, tmp_path):
    output_file = tmp_path / "output.jsonl"
    new_dataset, log_messages = prepare_dataset(sample_dataset, str(output_file), eval_n_limit=None)
    assert len(new_dataset) == 5
    assert not os.path.exists(output_file)
    assert any("Writing evaluation output to" in msg for msg in log_messages)

def test_prepare_dataset_resume(sample_dataset, tmp_path):
    output_file = tmp_path / "output.jsonl"
    
    # Create an existing output file with some finished instances
    with open(output_file, 'w') as f:
        f.write('{"instance_id": "1", "result": "done"}\n')
        f.write('{"instance_id": "3", "result": "done"}\n')
    
    new_dataset, log_messages = prepare_dataset(sample_dataset, str(output_file), eval_n_limit=None)
    
    assert len(new_dataset) == 3
    assert set(new_dataset['instance_id']) == {'2', '4', '5'}
    assert any("Resuming from where we left off" in msg for msg in log_messages)

def test_prepare_dataset_eval_n_limit(sample_dataset, tmp_path):
    output_file = tmp_path / "output.jsonl"
    new_dataset, log_messages = prepare_dataset(sample_dataset, str(output_file), eval_n_limit=3)
    assert len(new_dataset) == 3
    assert any("Limiting evaluation to first 3 instances" in msg for msg in log_messages)
