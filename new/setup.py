#!/usr/bin/env python3
"""
Setup script for TrOCR Table Extraction Project
Run this first on a new system to set up the environment
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_environment():
    """Check if all required packages are available"""
    required_packages = [
        'torch', 'torchvision', 'transformers', 
        'datasets', 'PIL', 'numpy', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def main():
    print("TrOCR Table Extraction Project Setup")
    print("=" * 40)
    
    # Check current environment
    print("\nChecking current environment...")
    if check_environment():
        print("\n✅ All packages are already installed!")
    else:
        print("\n📦 Installing missing packages...")
        if not install_requirements():
            print("\n❌ Setup failed. Please install packages manually:")
            print("pip install -r requirements.txt")
            sys.exit(1)
    
    # Create necessary directories
    print("\nCreating directories...")
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./outputs", exist_ok=True)
    print("✅ Directories created")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python DownloadandPreprocess.py")
    print("2. This will download and preprocess the PubTabNet dataset")
    print("3. The script runs in test mode by default (fast)")
    print("4. Change test_mode=False for full dataset processing")

if __name__ == "__main__":
    main()
