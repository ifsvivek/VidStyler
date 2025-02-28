import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import app
from app.app import create_ui

if __name__ == "__main__":
    app = create_ui()
    # Launch with larger default height and with sharing enabled
    app.launch(height=900, share=True)
