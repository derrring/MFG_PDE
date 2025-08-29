# Examples

This directory contains comprehensive examples and demonstrations of the MFG_PDE framework, organized by complexity and purpose.

## 📁 Directory Structure

### 🔰 [Basic Examples](basic/)
Simple, single-concept demonstrations perfect for getting started:

- **[particle_collocation_mfg_example.py](basic/particle_collocation_mfg_example.py)** - Basic MFG problem solving
- **[simple_logging_demo.py](basic/simple_logging_demo.py)** - Logging system introduction  
- **[mathematical_visualization_example.py](basic/mathematical_visualization_example.py)** - Mathematical plotting
- **[logging_integration_example.py](basic/logging_integration_example.py)** - Logging integration patterns

### 🚀 [Advanced Examples](advanced/)
Complex, multi-concept demonstrations showing advanced features:

- **[advanced_visualization_example.py](advanced/advanced_visualization_example.py)** - Sophisticated plotting and analysis
- **[factory_patterns_example.py](advanced/factory_patterns_example.py)** - Advanced solver configuration
- **[interactive_research_notebook_example.py](advanced/interactive_research_notebook_example.py)** - Research workflow automation
- **[enhanced_logging_demo.py](advanced/enhanced_logging_demo.py)** - Advanced logging features
- **[progress_monitoring_example.py](advanced/progress_monitoring_example.py)** - Real-time progress tracking
- **[retrofit_solver_logging.py](advanced/retrofit_solver_logging.py)** - Existing code integration
- **[logging_analysis_and_demo.py](advanced/logging_analysis_and_demo.py)** - Log analysis and monitoring

### 📓 [Notebooks](notebooks/)
Jupyter notebook demonstrations and interactive tutorials:

- **[working_demo/](notebooks/working_demo/)** - Complete working Jupyter notebook with advanced graphics
- **[advanced_notebook_demo.py](notebooks/advanced_notebook_demo.py)** - Notebook generation system demo
- **[notebook_demo_simple.py](notebooks/notebook_demo_simple.py)** - Simple notebook creation
- **[working_notebook_demo.py](notebooks/working_notebook_demo.py)** - Guaranteed working notebook generator

### 📚 [Tutorials](tutorials/)
Step-by-step guides and learning materials (planned for future expansion).

## 🎯 Getting Started

### For Beginners
1. Start with **[Basic Examples](basic/)** to understand core concepts
2. Try the **[working Jupyter notebook](notebooks/working_demo/)** for interactive learning
3. Progress to **[Advanced Examples](advanced/)** for sophisticated features

### For Researchers
1. Explore **[Notebooks](notebooks/)** for interactive analysis tools
2. Check **[Advanced Examples](advanced/)** for research workflow patterns
3. Use examples as templates for your own research

### For Developers
1. Study **[Advanced Examples](advanced/)** for integration patterns
2. Review **[Factory Patterns](advanced/factory_patterns_example.py)** for extensibility
3. Examine **[Logging Integration](basic/logging_integration_example.py)** for best practices

## 🔧 Requirements

### Basic Requirements
```bash
pip install numpy scipy matplotlib
```

### Full Features (including interactive notebooks)
```bash
pip install numpy scipy matplotlib plotly jupyter nbformat
```

### Installation from Source
```bash
git clone https://github.com/derrring/MFG_PDE.git
cd MFG_PDE
pip install -e .
```

## 🚀 Running Examples

### Python Scripts
```bash
# Basic example
python examples/basic/particle_collocation_mfg_example.py

# Advanced example  
python examples/advanced/advanced_visualization_example.py
```

### Jupyter Notebooks
```bash
# Start Jupyter and open notebook
jupyter lab examples/notebooks/working_demo/MFG_Working_Demo.ipynb

# Or run directly as Python
python examples/notebooks/working_demo/MFG_Working_Demo.py
```

## 📊 Example Categories

| Category | Files | Purpose | Complexity |
|----------|-------|---------|------------|
| **Basic** | 4 | Learning fundamentals | ⭐ |
| **Advanced** | 7 | Complex workflows | ⭐⭐⭐ |
| **Notebooks** | 4 | Interactive analysis | ⭐⭐ |
| **Tutorials** | 0+ | Step-by-step guides | ⭐ |

## 🔍 Finding the Right Example

- **New to MFG_PDE?** → Start with `basic/particle_collocation_mfg_example.py`
- **Want interactive plots?** → Try `notebooks/working_demo/`
- **Need logging?** → Check `basic/simple_logging_demo.py`
- **Research workflows?** → Explore `advanced/interactive_research_notebook_example.py`
- **Visualization focus?** → See `advanced/advanced_visualization_example.py`

## 📞 Help and Support

- **Documentation**: See [../docs/](../docs/) for comprehensive guides
- **Issues**: Report problems in the GitHub issue tracker
- **Questions**: Check [../docs/user/](../docs/user/) for user guides and troubleshooting

---

*Examples last updated: 2025-07-26*
