#!/usr/bin/env python3
"""
Start MLflow UI for the fraud detection system.
This script starts the MLflow tracking server with the appropriate configuration.
"""

import subprocess
import sys
import os
from pathlib import Path

def start_mlflow_ui():
    """Start MLflow UI with proper configuration."""
    
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    
    # Set up MLflow tracking directory
    mlruns_dir = project_root / "mlruns"
    mlruns_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("Starting MLflow UI for Fraud Detection System")
    print("="*60)
    print(f"Project root: {project_root}")
    print(f"MLruns directory: {mlruns_dir}")
    print()
    print("MLflow UI will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*60)
    
    try:
        # Start MLflow UI
        subprocess.run([
            "mlflow", "ui",
            "--backend-store-uri", f"sqlite:///{mlruns_dir}/mlflow.db",
            "--default-artifact-root", str(mlruns_dir),
            "--host", "0.0.0.0",
            "--port", "5000"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n\nMLflow UI stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\nError starting MLflow UI: {e}")
        print("Make sure MLflow is installed: pip install mlflow")
        sys.exit(1)
    except FileNotFoundError:
        print("\nMLflow not found. Please install MLflow first:")
        print("pip install mlflow")
        sys.exit(1)

if __name__ == "__main__":
    start_mlflow_ui()
