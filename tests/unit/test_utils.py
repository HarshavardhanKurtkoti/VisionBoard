import os
import numpy as np
import pytest
from visionboard.utils.main_utils.utils import (
    read_yaml_file,
    write_yaml_file,
    save_numpy_array,
    load_numpy_array,
    save_object,
    load_object,
    create_directories,
    get_size
)

def test_yaml_read_write(temp_test_dir):
    yaml_path = os.path.join(temp_test_dir, "test.yaml")
    data = {"project": "VisionBoard", "nc": 1, "classes": ["signboard"]}
    
    write_yaml_file(yaml_path, data)
    assert os.path.exists(yaml_path)
    
    loaded = read_yaml_file(yaml_path)
    assert loaded["project"] == "VisionBoard"
    assert loaded["classes"] == ["signboard"]

def test_numpy_save_load(temp_test_dir):
    arr_path = os.path.join(temp_test_dir, "test.npy")
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    
    save_numpy_array(arr_path, arr)
    assert os.path.exists(arr_path)
    
    loaded = load_numpy_array(arr_path)
    assert np.array_equal(arr, loaded)

def test_object_save_load(temp_test_dir):
    obj_path = os.path.join(temp_test_dir, "test.pkl")
    data = {"numbers": [1, 2, 3], "name": "test"}
    
    save_object(obj_path, data)
    assert os.path.exists(obj_path)
    
    loaded = load_object(obj_path)
    assert loaded == data

def test_create_directories_and_get_size(temp_test_dir):
    dir1 = os.path.join(temp_test_dir, "nested", "dir1")
    dir2 = os.path.join(temp_test_dir, "nested", "dir2")
    
    create_directories([dir1, dir2])
    assert os.path.isdir(dir1)
    assert os.path.isdir(dir2)
    
    size_str = get_size(temp_test_dir)
    assert "B" in size_str or "KB" in size_str
