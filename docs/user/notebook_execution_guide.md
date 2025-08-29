# Jupyter Notebook Execution Guide

## ✅ Working Notebook Available

**Location**: `working_demo/MFG_Working_Demo.ipynb`  
**Status**: ✅ Tested and guaranteed to work  
**Size**: 9.1 KB  

## 🚀 How to Execute

### Option 1: Jupyter Lab (Recommended)
```bash
jupyter lab working_demo/MFG_Working_Demo.ipynb
```

### Option 2: Jupyter Notebook (Classic)
```bash
jupyter notebook working_demo/MFG_Working_Demo.ipynb
```

### Option 3: Direct Python Execution (Testing)
```bash
python working_demo/MFG_Working_Demo.py
```

## 🔧 Troubleshooting Common Issues

### Issue 1: "jupyter: command not found"
**Solution**: Install Jupyter
```bash
pip install jupyter
# OR
conda install jupyter
```

### Issue 2: "No module named 'plotly'"
**Solution**: The notebook has matplotlib fallback
```bash
# Optional - for interactive plots
pip install plotly
```

### Issue 3: Plots don't show
**Solutions**:
1. **Restart kernel**: Kernel → Restart & Run All
2. **Check backend**: 
   ```python
   %matplotlib inline
   import matplotlib.pyplot as plt
   plt.ion()  # Interactive mode
   ```
3. **Clear output**: Cell → All Output → Clear

### Issue 4: Kernel won't start
**Solutions**:
1. **Check Python environment**:
   ```bash
   which python
   python --version
   ```
2. **Reinstall kernel**:
   ```bash
   python -m ipykernel install --user
   ```
3. **Check Jupyter installation**:
   ```bash
   jupyter --version
   ```

### Issue 5: Import errors
**Solution**: The notebook handles this gracefully with fallbacks
- If `plotly` missing → Uses matplotlib
- If `numpy` missing → Install: `pip install numpy matplotlib`

## 📊 What the Notebook Contains

### ✅ Working Features
- **Error handling** for missing dependencies
- **Matplotlib fallback** if Plotly unavailable
- **Self-contained data generation** (no external files needed)
- **Interactive plots** (if Plotly available)
- **Mathematical analysis** with LaTeX
- **Export instructions** for HTML/PDF

### 🎯 Generated Content
- **50×30 solution grid** with realistic MFG data
- **Value function u(t,x)** evolution
- **Population density m(t,x)** dynamics
- **Convergence analysis** (20 iterations)
- **Mass conservation** verification

## 🔄 Alternative Execution Methods

### Method 1: Convert to HTML (Non-interactive)
```bash
cd working_demo
jupyter nbconvert --to html --execute MFG_Working_Demo.ipynb
# Opens: MFG_Working_Demo.html in browser
```

### Method 2: Convert to PDF
```bash
cd working_demo
jupyter nbconvert --to pdf MFG_Working_Demo.ipynb
# Requires: LaTeX installation
```

### Method 3: Run as Python Script
```bash
cd working_demo
python MFG_Working_Demo.py
```

## 🎛️ Customization Options

### Change Grid Resolution
```python
# In data generation cell, modify:
nx, nt = 100, 60  # Higher resolution
nx, nt = 25, 15   # Lower resolution (faster)
```

### Switch to Static Plots Only
```python
# In visualization cell, force matplotlib:
PLOTLY_AVAILABLE = False
```

### Add More Analysis
```python
# Add new cells with additional analysis:
print("Custom analysis:")
print(f"Max gradient: {np.max(np.gradient(u, axis=1)):.3f}")
```

## 🐛 Still Having Issues?

### Check System Requirements
1. **Python**: 3.7+
2. **Required packages**: `numpy`, `matplotlib`
3. **Optional packages**: `plotly`, `nbformat`
4. **Jupyter**: Latest version recommended

### Minimal Installation
```bash
pip install jupyter numpy matplotlib
# This is sufficient to run the notebook
```

### Full Installation (Recommended)
```bash
pip install jupyter numpy matplotlib plotly nbformat
# Enables all features including interactive plots
```

### Verify Installation
```bash
python -c "import numpy, matplotlib.pyplot; print('✅ Core packages OK')"
python -c "import plotly; print('✅ Plotly OK')" 2>/dev/null || echo "⚠️ Plotly missing (fallback available)"
```

## 🎯 Expected Output

When the notebook executes successfully, you should see:
- ✅ Dependency check messages
- 📊 Data generation confirmation
- 🎨 Visualization creation
- 📋 Numerical summary
- Interactive plots (if Plotly available) or static plots (matplotlib fallback)

## 📞 Support

If you still can't execute the notebook:
1. Check the Python script version works: `python working_demo/MFG_Working_Demo.py`
2. Verify Jupyter installation: `jupyter --version`
3. Try the minimal dependencies: `pip install jupyter numpy matplotlib`

The notebook is designed to be **foolproof** with comprehensive error handling and fallbacks. It should work in any standard Python environment with basic scientific packages.
