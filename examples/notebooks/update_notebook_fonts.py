#!/usr/bin/env python3
"""Update the notebook with clean font configuration."""

import json
from pathlib import Path

def update_notebook_with_clean_fonts():
    """Update notebook to include clean font configuration."""
    
    notebook_path = Path("results/working_demo/mfg_demonstration.ipynb")
    
    # Read existing notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Add font configuration cell at the beginning
    font_config_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Configure matplotlib fonts to avoid warnings\n",
            "import matplotlib as mpl\n",
            "\n",
            "mpl.rcParams.update({\n",
            "    'font.family': 'sans-serif',\n",
            "    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica'],\n",
            "    'mathtext.fontset': 'dejavusans',\n",
            "    'text.usetex': False,\n",
            "    'axes.formatter.use_mathtext': True\n",
            "})\n",
            "\n",
            "# Clear font cache\n",
            "try:\n",
            "    mpl.font_manager._rebuild()\n",
            "except:\n",
            "    pass\n",
            "\n",
            "print(\"🎨 Matplotlib fonts configured for clean output\")"
        ]
    }
    
    # Insert font configuration after the first import cell
    notebook['cells'].insert(1, font_config_cell)
    
    # Save updated notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"📓 Updated notebook with clean font configuration: {notebook_path}")

if __name__ == "__main__":
    update_notebook_with_clean_fonts()