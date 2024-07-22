import os
import sys
import subprocess
import pkg_resources

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Function to upgrade a package
def upgrade_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])

# Upgrade packages from requirements.txt
requirements = [
    "git+https://github.com/huggingface/diffusers",
    "transformers",
    "accelerate"
]

for requirement in requirements:
    try:
        upgrade_package(requirement)
    except Exception as e:
        print(f"Failed to upgrade {requirement}: {str(e)}")

# Import after upgrading packages
from .lumina_diffusers_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
