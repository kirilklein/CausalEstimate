import importlib
import os
import pkgutil


def import_all_estimators():
    package_dir = os.path.join(os.path.dirname(__file__), "..", "estimators")
    for _, module_name, _ in pkgutil.iter_modules([package_dir]):
        importlib.import_module(f"CausalEstimate.estimators.{module_name}")
