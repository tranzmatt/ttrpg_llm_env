# Environment Setup Script for TTRPG LLM Training
# Updated for Ubuntu 24.04 with Python 3.12 (or Ubuntu 22.04 with Python 3.10)

import subprocess
import sys
import os
import platform

def get_python_version():
    """Detect the appropriate Python version based on Ubuntu version"""
    try:
        # Get Ubuntu version
        with open('/etc/os-release', 'r') as f:
            os_info = f.read()
        
        if 'VERSION_ID="24.04"' in os_info:
            return "python3.12", "3.12"
        elif 'VERSION_ID="22.04"' in os_info:
            return "python3.10", "3.10"
        else:
            # Default to system python3
            result = subprocess.run("python3 --version", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                version_output = result.stdout.strip()
                if "3.12" in version_output:
                    return "python3.12", "3.12"
                elif "3.10" in version_output:
                    return "python3.10", "3.10"
                elif "3.11" in version_output:
                    return "python3.11", "3.11"
            
            # Final fallback
            return "python3", "default"
            
    except Exception as e:
        print(f"Warning: Could not detect Ubuntu version: {e}")
        return "python3", "default"

def run_command(command, description, check_exit_code=True):
    """Run a shell command and handle errors"""
    print(f"\n{'='*50}")
    print(f"🔧 {description}")
    print('='*50)
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            print("Output:")
            print(result.stdout)
        
        if result.stderr and "warning" not in result.stderr.lower():
            print("Warnings/Errors:")
            print(result.stderr)
        
        if check_exit_code and result.returncode != 0:
            print(f"❌ Command failed with exit code: {result.returncode}")
            return False
        else:
            print("✅ Success!")
            return True
            
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def check_system_info():
    """Check and display system information"""
    print("\n🖥️  System Information:")
    
    try:
        # Ubuntu version
        with open('/etc/os-release', 'r') as f:
            for line in f:
                if line.startswith('PRETTY_NAME='):
                    ubuntu_version = line.split('=')[1].strip().strip('"')
                    print(f"OS: {ubuntu_version}")
                    break
    except:
        print("OS: Unknown Linux")
    
    # Python version
    python_cmd, python_ver = get_python_version()
    result = subprocess.run(f"{python_cmd} --version", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Python: {result.stdout.strip()}")
    else:
        print(f"Python: {python_cmd} (recommended: {python_ver})")
    
    return python_cmd, python_ver

def check_cuda_installation():
    """Check if CUDA is properly installed"""
    print("\n🔍 Checking CUDA installation...")
    
    # Check nvidia-smi
    result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ nvidia-smi not found. Please install NVIDIA drivers first.")
        return False
    
    print("✅ NVIDIA drivers found")
    print(result.stdout)
    
    # Check nvcc
    result = subprocess.run("nvcc --version", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("⚠️  CUDA compiler (nvcc) not found. Will install CUDA toolkit.")
        return False
    
    print("✅ CUDA toolkit found")
    print(result.stdout)
    return True

def install_cuda_toolkit():
    """Install CUDA 12.8 toolkit"""
    print("\n📦 Installing CUDA 12.8 toolkit...")
    
    # Detect Ubuntu version for correct repository
    ubuntu_version = "ubuntu2404"  # Default to 24.04
    try:
        with open('/etc/os-release', 'r') as f:
            os_info = f.read()
            if 'VERSION_ID="22.04"' in os_info:
                ubuntu_version = "ubuntu2204"
    except:
        pass
    
    commands = [
        f"wget https://developer.download.nvidia.com/compute/cuda/repos/{ubuntu_version}/x86_64/cuda-keyring_1.1-1_all.deb",
        "sudo dpkg -i cuda-keyring_1.1-1_all.deb", 
        "sudo apt-get update",
        "sudo apt-get -y install cuda-toolkit-12-8"
    ]
    
    for cmd in commands:
        if not run_command(cmd, f"Running: {cmd}"):
            return False
    
    # Set environment variables
    cuda_path = "/usr/local/cuda-12.8"
    bashrc_additions = f"""
# CUDA 12.8 Environment Variables
export CUDA_HOME={cuda_path}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
"""
    
    with open(os.path.expanduser("~/.bashrc"), "a") as f:
        f.write(bashrc_additions)
    
    print("✅ CUDA environment variables added to ~/.bashrc")
    print("⚠️  Please run 'source ~/.bashrc' or restart your terminal")
    
    return True

def setup_python_environment(python_cmd, python_ver):
    """Set up Python environment for ML training"""
    env_name = "ttrpg_llm_env"
    
    print(f"\n🐍 Setting up Python environment: {env_name}")
    print(f"Using Python: {python_cmd} ({python_ver})")
    
    # Check if the specified Python version is available
    result = subprocess.run(f"{python_cmd} --version", shell=True, capture_output=True)
    if result.returncode != 0:
        print(f"Installing {python_cmd}...")
        
        # Install the appropriate Python version and dev packages
        if python_ver == "3.12":
            install_cmd = "sudo apt update && sudo apt install -y python3.12 python3.12-venv python3.12-dev python3-pip"
        elif python_ver == "3.10":
            install_cmd = "sudo apt update && sudo apt install -y python3.10 python3.10-venv python3.10-dev python3-pip"
        else:
            install_cmd = "sudo apt update && sudo apt install -y python3 python3-venv python3-dev python3-pip"
        
        if not run_command(install_cmd, f"Installing {python_cmd}"):
            return False
    
    # Create virtual environment
    if not run_command(f"{python_cmd} -m venv {env_name}", "Creating virtual environment"):
        return False
    
    print(f"✅ Virtual environment created: {env_name}")
    print(f"To activate: source {env_name}/bin/activate")
    
    return True

def install_python_packages():
    """Install required Python packages"""
    print("\n📦 Installing Python packages...")
    
    # Check if we're in a virtual environment
    if not os.environ.get('VIRTUAL_ENV'):
        print("⚠️  Virtual environment not activated!")
        print("Please run: source ttrpg_llm_env/bin/activate")
        return False
    
    print(f"Virtual environment: {os.environ.get('VIRTUAL_ENV')}")
    
    # Upgrade pip first
    run_command("pip install --upgrade pip", "Upgrading pip")
    
    # Install packages in order of dependency
    package_groups = [
        # Core dependencies first
        ["numpy", "packaging"],
        
        # PyTorch with CUDA support
        ["torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"],
        
        # Transformers ecosystem
        ['"transformers>=4.40.0"', "tokenizers"],
        
        # Training dependencies
        ["datasets", "accelerate", '"peft>=0.10.0"', "bitsandbytes"],
        
        # Training framework
        ["trl"],
        
        # PDF processing
        ["PyMuPDF", "pypdf2", "pdfplumber"],
        
        # Utilities
        ["pandas", "tqdm", "scikit-learn"],
        
        # Unsloth (install last as it may have conflicts)
        ['"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"']
    ]
    
    for group in package_groups:
        for package in group:
            if not run_command(f"pip install {package}", f"Installing {package}"):
                print(f"⚠️  Failed to install {package}, continuing...")
    
    return True

def test_installation():
    """Test if everything is installed correctly"""
    print("\n🧪 Testing installation...")
    
    # Test packages using subprocess to avoid import issues
    test_commands = [
        ("PyTorch", "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'No CUDA')"),
        ("Transformers", "import transformers; print(f'Transformers: {transformers.__version__}')"),
        ("Datasets", "import datasets; print(f'Datasets: {datasets.__version__}')"),
        ("Accelerate", "import accelerate; print(f'Accelerate: {accelerate.__version__}')"),
        ("BitsAndBytes", "import bitsandbytes; print('BitsAndBytes: OK')"),
        ("PEFT", "import peft; print(f'PEFT: {peft.__version__}')"),
        ("TRL", "import trl; print(f'TRL: {trl.__version__}')"),
        ("PyMuPDF", "import fitz; print('PyMuPDF: OK')"),
    ]
    
    all_good = True
    for name, test_code in test_commands:
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {result.stdout.strip()}")
        else:
            print(f"❌ {name} failed: {result.stderr.strip()}")
            all_good = False
    
    # Test Unsloth separately (may need restart)
    result = subprocess.run([sys.executable, "-c", "import unsloth; print('Unsloth: OK')"], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Unsloth: OK")
    else:
        print("⚠️  Unsloth not available (may need terminal restart)")
    
    # Test GPU if available
    if all_good:
        result = subprocess.run([sys.executable, "-c", 
            """
import torch
if torch.cuda.is_available():
    print(f'🔥 GPU: {torch.cuda.get_device_name()}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print('   Ready for 7B model training with QLoRA!')
else:
    print('⚠️  No CUDA GPU detected - will use CPU (much slower)')
            """], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
    
    return all_good

def create_project_structure():
    """Create project directory structure"""
    print("\n📁 Creating project structure...")
    
    directories = [
        "./ttrpg_pdfs",           # Place your PDF files here
        "./trained_models",       # Trained models will be saved here
        "./datasets",            # Generated datasets
        "./logs",               # Training logs
        "./exports"             # Exported models
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")
    
    # Create README with system-specific instructions
    python_cmd, python_ver = get_python_version()
    readme_content = f"""# TTRPG LLM Project

## System Information
- Python: {python_cmd} ({python_ver})
- Environment: ttrpg_llm_env

## Directory Structure
- `ttrpg_pdfs/` - Place your TTRPG PDF files here
- `trained_models/` - Trained models are saved here
- `datasets/` - Generated training datasets
- `logs/` - Training logs and outputs
- `exports/` - Exported models in different formats

## Quick Start
```bash
# 1. Setup environment (if not done already)
python3 setup_environment_updated.py
source ttrpg_llm_env/bin/activate

# 2. Place PDFs in ttrpg_pdfs/ directory

# 3. Run pipeline
python pdf_extractor.py
python train_ttrpg_llm.py
python gm_inference.py
```

## Scripts
- `setup_environment_updated.py` - Environment setup (run first)
- `pdf_extractor.py` - Extract text from PDFs
- `train_ttrpg_llm.py` - Train the LLM
- `gm_inference.py` - Use the trained model

## Troubleshooting
- If you get CUDA out of memory errors, reduce batch_size in train_ttrpg_llm.py
- For Ubuntu 24.04: Uses Python 3.12
- For Ubuntu 22.04: Uses Python 3.10
- Monitor GPU with: nvidia-smi
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Created README.md")
    return True

def main():
    """Main setup function"""
    print("🎲 TTRPG LLM Environment Setup")
    print("=" * 50)
    
    # Check system info
    python_cmd, python_ver = check_system_info()
    
    print("\nThis script will set up your environment for training a TTRPG Game Master LLM")
    print("Target GPU: RTX 3000 series (12-16GB VRAM)")
    print("=" * 50)
    
    # Check if running as root
    if os.geteuid() == 0:
        print("⚠️  Don't run this script as root! Run as regular user.")
        sys.exit(1)
    
    success_steps = []
    
    # Step 1: Check CUDA
    if check_cuda_installation():
        success_steps.append("CUDA check")
    else:
        print("\n❓ Would you like to install CUDA 12.8? (y/n)")
        response = input().strip().lower()
        if response.startswith('y'):
            if install_cuda_toolkit():
                success_steps.append("CUDA installation")
                print("\n⚠️  Please restart your terminal and re-run this script to continue")
                return
        else:
            print("⚠️  CUDA is required for efficient GPU training")
    
    # Step 2: Python environment
    if setup_python_environment(python_cmd, python_ver):
        success_steps.append("Python environment")
    
    # Step 3: Check if virtual env is active
    if not os.environ.get('VIRTUAL_ENV'):
        print("\n⚠️  Please activate the virtual environment and re-run:")
        print("source ttrpg_llm_env/bin/activate")
        print("python3 setup_environment_updated.py")
        return
    
    # Step 4: Install packages
    if install_python_packages():
        success_steps.append("Package installation")
    
    # Step 5: Test installation
    if test_installation():
        success_steps.append("Installation test")
    
    # Step 6: Create project structure
    if create_project_structure():
        success_steps.append("Project structure")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    print("✅ Completed steps:", ", ".join(success_steps))
    
    print(f"\n📋 Next Steps (using {python_cmd}):")
    print("1. Place your TTRPG PDF files in ./ttrpg_pdfs/")
    print("2. Run: python pdf_extractor.py")
    print("3. Run: python train_ttrpg_llm.py") 
    print("4. Run: python gm_inference.py")
    
    print("\n💡 Tips:")
    print("- Start with a small dataset to test everything works")
    print("- Monitor GPU memory with nvidia-smi during training")
    print("- Use batch_size=1 if you get out-of-memory errors")
    print(f"- Your Python version ({python_ver}) is compatible with all ML libraries")

if __name__ == "__main__":
    main()
