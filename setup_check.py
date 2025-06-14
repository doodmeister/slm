#!/usr/bin/env python3
"""
Setup validation script for the SLM project.
Ensures all dependencies are installed and the environment is properly configured.
"""

import sys
import importlib
import subprocess
from pathlib import Path
from typing import List, Tuple

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_dependencies() -> Tuple[List[str], List[str]]:
    """Check if all required dependencies are installed."""
    required_packages = [
        'torch',
        'transformers',
        'numpy',
        'matplotlib',
        'pydantic',
        'click',
        'rich',
        'loguru',
        'yaml',
        'tqdm',
        'psutil'
    ]
    
    installed = []
    missing = []
    
    for package in required_packages:
        try:
            if package == 'yaml':
                importlib.import_module('yaml')
            else:
                importlib.import_module(package)
            installed.append(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} - Missing")
    
    return installed, missing

def check_dev_dependencies() -> Tuple[List[str], List[str]]:
    """Check if development dependencies are installed."""
    dev_packages = [
        'pytest',
        'flake8',
        'mypy',
        'black',
        'isort'
    ]
    
    installed = []
    missing = []
    
    for package in dev_packages:
        try:
            importlib.import_module(package)
            installed.append(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} - Missing (dev dependency)")
    
    return installed, missing

def check_project_structure() -> bool:
    """Check if the project structure is correct."""
    project_root = Path(__file__).parent
    
    required_files = [
        'slm/__init__.py',
        'slm/config.py',
        'slm/cli.py',
        'slm/core/models.py',
        'slm/core/data.py',
        'slm/core/trainer.py',
        'slm/core/generator.py',
        'tests/test_slm.py',
        'pyproject.toml',
        'config.yaml'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path} - Missing")
    
    return len(missing_files) == 0

def install_missing_dependencies(missing: List[str]) -> bool:
    """Install missing dependencies."""
    if not missing:
        return True
    
    print(f"\n📦 Installing missing dependencies: {', '.join(missing)}")
    
    try:
        # Try to install with pip
        cmd = [sys.executable, '-m', 'pip', 'install'] + missing
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print(f"Error output: {e.stderr}")
        return False

def run_basic_tests() -> bool:
    """Run basic import tests."""
    print("\n🧪 Running basic import tests...")
    
    try:
        # Test core imports
        from slm.config import Config
        
        print("✅ All core modules import successfully")
        
        # Test configuration loading
        config = Config.from_yaml("config.yaml")
        print("✅ Configuration loading works")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    """Main validation function."""
    print("🔍 SLM Project Setup Validation")
    print("=" * 40)
    
    # Check Python version
    print("\n🐍 Checking Python version...")
    python_ok = check_python_version()
    
    # Check project structure
    print("\n📁 Checking project structure...")
    structure_ok = check_project_structure()
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    installed, missing = check_dependencies()
    
    # Check dev dependencies
    print("\n🛠️  Checking development dependencies...")
    dev_installed, dev_missing = check_dev_dependencies()
    
    # Install missing dependencies if needed
    if missing:
        install_success = install_missing_dependencies(missing)
        if not install_success:
            print("\n❌ Setup validation failed - could not install dependencies")
            return False
    
    # Run basic tests
    if python_ok and structure_ok and not missing:
        test_ok = run_basic_tests()
    else:
        test_ok = False
    
    # Summary
    print("\n" + "=" * 40)
    print("📋 Setup Validation Summary")
    print("=" * 40)
    
    if python_ok and structure_ok and not missing and test_ok:
        print("✅ Setup validation successful!")
        print("🚀 Your SLM project is ready to use!")
        print("\nNext steps:")
        print("  1. Train a model: python -m slm.cli train")
        print("  2. Generate text: python -m slm.cli generate")
        print("  3. Open GUI: python -m slm.gui")
        print("  4. Run tests: pytest tests/")
    else:
        print("❌ Setup validation failed")
        if not python_ok:
            print("  - Upgrade to Python 3.8+")
        if not structure_ok:
            print("  - Fix project structure")
        if missing:
            print(f"  - Install missing dependencies: {', '.join(missing)}")
        if not test_ok:
            print("  - Fix import errors")
    
    return python_ok and structure_ok and not missing and test_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
