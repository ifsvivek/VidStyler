import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Create a temp directory for file operations
os.makedirs(os.path.join(project_root, "temp"), exist_ok=True)

# Import app
from app.app import create_ui

if __name__ == "__main__":
    app = create_ui()
    # Launch with larger default height and with sharing enabled
    app.launch(height=1080)
