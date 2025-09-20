#!/usr/bin/env python3
"""
Setup script for atom testing system
"""

import os
import sys
import shutil

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 7):
        issues.append("Python 3.7+ required")
    
    # Check required files
    required_files = [
        "operatorRAW.json",
        "credential.txt"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            issues.append(f"Missing required file: {file}")
    
    # Check cache files
    cache_files = [f for f in os.listdir('.') if f.startswith('data_fields_cache_') and f.endswith('.json')]
    if not cache_files:
        issues.append("No data_fields_cache_*.json files found")
    else:
        print(f"✅ Found {len(cache_files)} cache files")
    
    # Check if operatorRAW.json exists
    if not os.path.exists("operatorRAW.json"):
        print("⚠️ operatorRAW.json not found. Please copy from consultant-templates-api directory:")
        print("   copy ..\\consultant-templates-api\\operatorRAW.json .")
        issues.append("operatorRAW.json missing")
    else:
        print("✅ operatorRAW.json found")
    
    # Check credentials
    if not os.path.exists("credential.txt"):
        print("⚠️ credential.txt not found. Please create it with your WorldQuant Brain credentials:")
        print("   Format 1 (JSON): [\"username\", \"password\"]")
        print("   Format 2 (Two-line): username on line 1, password on line 2")
        print("   (Copy credential.example.txt to credential.txt and edit)")
        issues.append("credential.txt missing")
    else:
        print("✅ credential.txt found")
    
    if issues:
        print("\n❌ Setup issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("\n✅ All requirements met!")
        return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_sample_credential():
    """Create sample credential file if it doesn't exist"""
    if not os.path.exists("credential.txt") and os.path.exists("credential.example.txt"):
        print("\n📝 Creating sample credential file...")
        shutil.copy("credential.example.txt", "credential.txt")
        print("✅ Created credential.txt from example")
        print("⚠️ Please edit credential.txt with your actual WorldQuant Brain credentials")

def main():
    """Main setup function"""
    print("🚀 Setting up Atom Testing System...")
    print("="*50)
    
    # Create sample credential file
    create_sample_credential()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Setup incomplete. Please resolve the issues above.")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies.")
        sys.exit(1)
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Edit credential.txt with your WorldQuant Brain credentials (JSON or two-line format)")
    print("2. Run: python run_atom_tests.py")
    print("3. Analyze results: python analyze_atom_results.py")
    
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
