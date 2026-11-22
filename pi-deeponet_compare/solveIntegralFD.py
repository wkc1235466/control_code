import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "data_generation" / "solveIntegralFD.py"
_spec = importlib.util.spec_from_file_location("_data_generation_solveIntegralFD", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
solveIntegralFD = _mod.solveIntegralFD
boundary_derivatives = _mod.boundary_derivatives
