import importlib
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_import_src():
    m = importlib.import_module("src")
    if not hasattr(m, "__all__"):
        raise AssertionError("src import failed")

if __name__ == "__main__":
    test_import_src()
    print("test_init passed")