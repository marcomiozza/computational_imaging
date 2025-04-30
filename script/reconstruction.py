import importlib
import traceback
import sys

# Aggiunge IPPy al path se non lo trova
sys.path.append("/content/COMPUTATIONAL_IMAGING")

modules_to_check = {
    "torch": "PyTorch",
    "numpy": "NumPy",
    "scipy": "SciPy",
    "matplotlib": "Matplotlib",
    "skimage": "scikit-image",
    "numba": "Numba",
    "astra": "ASTRA Toolbox",
    "IPPy": "IPPy (root)",
    "IPPy.operators": "IPPy.operators",
    "IPPy.solvers": "IPPy.solvers",
    "IPPy.data": "IPPy.data",
    "IPPy._utilities": "IPPy._utilities"
}


def check_import(module_name, label):
    try:
        importlib.import_module(module_name)
        print(f"[OK] {label}")
    except Exception as e:
        print(f"[FAIL] {label}")
        print("       ", traceback.format_exc().splitlines()[-1])


if __name__ == "__main__":
    print("=== Diagnostica modulo ===")
    for mod, label in modules_to_check.items():
        check_import(mod, label)
