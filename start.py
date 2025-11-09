# -*- coding: utf-8 -*-

"""
Metal Multi-Axial Fatigue Life Prediction System - Quick Start Script
"""

import os
import sys
import subprocess

if __name__ == "__main__":
    # Get the directory where the current script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Command line arguments
    cmd_args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    # Build startup command
    cmd = [sys.executable, os.path.join(current_dir, "fatigue_prediction", "run.py"), "--debug"]
    cmd.extend(cmd_args)
    
    # Print prompt information
    print("Starting Metal Multi-Axial Fatigue Life Prediction System...")
    print(f"Command: {' '.join(cmd)}")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        # Start application
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServer stopped")
        sys.exit(0)
    except Exception as e:
        print(f"Failed to start: {e}")
        sys.exit(1) 