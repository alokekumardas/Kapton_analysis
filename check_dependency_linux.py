import sys
import subprocess
import pkg_resources
import os

def install_package(package, version=None):
    """Install or upgrade a package to the specified version."""
    if version:
        package_spec = f"{package}=={version}"
    else:
        package_spec = package
    
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", package_spec], check=True)

def check_python_version(required_version):
    """Check if the installed Python version matches the required version."""
    installed_version = sys.version.split()[0]
    if installed_version != required_version:
        print(f"Warning: Python version mismatch! Installed: {installed_version}, Required: {required_version}")
    else:
        print(f"Python version {installed_version} is correct.")

def check_and_install_packages(requirements):
    """Check and install missing or incorrect versions of required packages."""
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    for package, required_version in requirements.items():
        if package in installed_packages:
            installed_version = installed_packages[package]
            if installed_version != required_version:
                print(f"Updating {package} from {installed_version} to {required_version}")
                install_package(package, required_version)
            else:
                print(f"{package} is up to date ({installed_version})")
        else:
            print(f"Installing {package} {required_version}")
            install_package(package, required_version)

def install_system_dependencies():
    """Ensure system dependencies are installed."""
    print("Updating system packages...")
    subprocess.run(["sudo", "dnf", "update", "-y"], check=True)
    print("Installing system dependencies...")
    subprocess.run(["sudo", "dnf", "install", "-y", "python3-pip", "gcc", "python3-devel", "mariadb-devel", "postgresql-devel"], check=True)

if __name__ == "__main__":
    # Install system dependencies
    install_system_dependencies()
    
    # Check Python version
    check_python_version("3.9.0")
    
    # Define required packages and versions
    required_packages = {
        "mysql-connector": "2.2.9",
        "psycopg2": "2.9.6",
        "psycopg2-binary": "2.9.5",
        "PyQt5": "5.15.9",
        "PyYAML": "6.0",
        "matplotlib": "3.7.1",
        "numpy": "1.25.2",
        "Pillow": "9.5.0",
        "opencv-python": "4.8.1",
        "scipy": "1.11.1",
        "scikit-learn": "1.3.0",
        "paramiko": "3.1.0",
        "pandas": "2.1.3"
    }
    
    # Check and install required packages
    check_and_install_packages(required_packages)
    
    # Check built-in modules (no installation required)
    builtin_modules = ["subprocess", "array", "ntpath", "datetime", "shutil", "sys", "os", "glob", "math", "copy", "pathlib"]
    for module in builtin_modules:
        try:
            __import__(module)
            print(f"{module} is available.")
        except ImportError:
            print(f"Error: {module} is missing (should not happen for built-in modules).")
