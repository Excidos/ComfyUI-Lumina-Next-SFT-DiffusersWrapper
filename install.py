import subprocess
import sys
import os
import pkg_resources

def get_installed_version(package_name):
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None

def install_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    try:
        with open(requirements_path, 'r') as f:
            requirements = f.readlines()

        for requirement in requirements:
            if requirement.strip() and not requirement.startswith('#'):
                if requirement.startswith('git+'):
                    # For git repositories, check if the package is installed
                    package_name = requirement.split('/')[-1].split('@')[0]
                    if get_installed_version(package_name) is None:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", requirement.strip()])
                        print(f"Installed {requirement.strip()}")
                    else:
                        print(f"{package_name} is already installed")
                else:
                    package_name = requirement.split('==')[0].split('>=')[0].strip()
                    required_version = requirement.split('==')[1].strip() if '==' in requirement else None
                    
                    installed_version = get_installed_version(package_name)
                    
                    if installed_version is None:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", requirement.strip()])
                        print(f"Installed {requirement.strip()}")
                    elif required_version and installed_version != required_version:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", requirement.strip()])
                        print(f"Upgraded {package_name} from {installed_version} to {required_version}")
                    else:
                        print(f"{package_name} is already up to date ({installed_version})")
        
        print("Successfully checked and installed requirements for Lumina Diffusers Node")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    install_requirements()