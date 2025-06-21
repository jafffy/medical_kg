#!/usr/bin/env python3
"""
Command-line interface for SOAP Knowledge Graph Generator
"""

import sys
import os

def main():
    """Main CLI entry point"""
    # Add the scripts directory to path to import main.py
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'scripts')
    sys.path.insert(0, scripts_dir)
    
    # Import and run the main script
    try:
        from main import main as run_main
        run_main()
    except ImportError as e:
        print(f"Error importing main script: {e}")
        print("Please ensure the package is properly installed.")
        sys.exit(1)

if __name__ == "__main__":
    main()