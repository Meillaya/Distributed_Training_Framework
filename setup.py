import subprocess
import sys
import os
import shutil

def check_uv_installed():
    """Checks if uv is installed."""
    return shutil.which("uv") is not None

def setup_environment():
    """Creates a virtual environment and installs dependencies using uv."""
    if os.path.exists(".venv"):
        print("Removing old '.venv' directory.")
        shutil.rmtree(".venv")

    venv_dir = ".venv"
    print("Creating virtual environment with uv...")
    subprocess.check_call(["uv", "venv"])
    print(f"Virtual environment created at '{venv_dir}'.")

    print("Installing dependencies with uv...")
    subprocess.check_call(["uv", "pip", "install", "-r", "requirements.txt"])
    print("Dependencies installed.")

if __name__ == "__main__":
    if not check_uv_installed():
        print("Error: 'uv' is not installed or not in your PATH.", file=sys.stderr)
        print("Please install uv by following the instructions at https://github.com/astral-sh/uv", file=sys.stderr)
        sys.exit(1)
    
    setup_environment() 