import pkgutil
import importlib
import os

# Get the directory of the current module
package_dir = os.path.dirname(__file__)

# Dynamically import all modules in the estimators package
for _, module_name, _ in pkgutil.iter_modules([package_dir]):
    importlib.import_module(f"{__name__}.{module_name}")
