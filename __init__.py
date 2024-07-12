import os
import subprocess
import sys
import pkg_resources

def get_installed_version(package_name):
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None

def install_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        try:
            with open(requirements_path, 'r') as f:
                requirements = f.readlines()

            for requirement in requirements:
                if requirement.strip() and not requirement.startswith('#'):
                    if requirement.startswith('git+'):
                        package_name = requirement.split('/')[-1].split('@')[0]
                        if get_installed_version(package_name) is None:
                            subprocess.check_call([sys.executable, "-m", "pip", "install", requirement.strip()])
                            print(f"Installed {requirement.strip()}")
                        else:
                            print(f"{package_name} is already installed")
                    else:
                        package_name = requirement.split('==')[0].split('>=')[0].strip()
                        installed_version = get_installed_version(package_name)
                        
                        if installed_version is None:
                            subprocess.check_call([sys.executable, "-m", "pip", "install", requirement.strip()])
                            print(f"Installed {requirement.strip()}")
                        else:
                            print(f"{package_name} is already installed (version {installed_version})")
            
            print("Successfully checked and installed requirements for Lumina Diffusers Node")
        except subprocess.CalledProcessError as e:
            print(f"Error installing requirements: {e}")
    else:
        print("requirements.txt not found. Skipping installation.")

# Install requirements when the module is imported
install_requirements()

# Now import the node mappings
try:
    from .lumina_diffusers_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError as e:
    print(f"Error importing Lumina Diffusers Node: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']