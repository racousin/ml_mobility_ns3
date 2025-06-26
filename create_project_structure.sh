#!/bin/bash

# Create project structure for ml_mobility_ns3
echo "Creating project structure for ml_mobility_ns3..."

# Create root level files
touch pyproject.toml
echo "Created pyproject.toml"

# Create .gitignore
touch .gitignore
echo "Created .gitignore"

# Create notebooks directory and files
mkdir -p notebooks
touch notebooks/01_data_exploration.ipynb
touch notebooks/02_model_experiments.ipynb
echo "Created notebooks directory and files"

# Create ml_mobility_ns3 directory structure
mkdir -p ml_mobility_ns3/data
mkdir -p ml_mobility_ns3/models
mkdir -p ml_mobility_ns3/training
mkdir -p ml_mobility_ns3/inference
mkdir -p ml_mobility_ns3/utils

# Create __init__.py files
touch ml_mobility_ns3/__init__.py
touch ml_mobility_ns3/data/__init__.py
touch ml_mobility_ns3/data/loader.py
touch ml_mobility_ns3/data/preprocessor.py
touch ml_mobility_ns3/models/__init__.py
touch ml_mobility_ns3/models/vae.py
touch ml_mobility_ns3/training/__init__.py
touch ml_mobility_ns3/training/trainer.py
touch ml_mobility_ns3/inference/__init__.py
touch ml_mobility_ns3/inference/generator.py
touch ml_mobility_ns3/utils/__init__.py
touch ml_mobility_ns3/utils/visualization.py
echo "Created ml_mobility_ns3 directory structure and files"

# Create tests directory and files
mkdir -p tests
touch tests/__init__.py
touch tests/test_data_loader.py
echo "Created tests directory and files"

echo "Project structure created successfully!"
echo ""
echo "Generated structure:"
echo "├── pyproject.toml"
echo "├── README.md"
echo "├── .gitignore"
echo "├── notebooks/"
echo "│   ├── 01_data_exploration.ipynb"
echo "│   └── 02_model_experiments.ipynb"
echo "├── ml_mobility_ns3/"
echo "│   ├── __init__.py"
echo "│   ├── data/"
echo "│   │   ├── __init__.py"
echo "│   │   ├── loader.py"
echo "│   │   └── preprocessor.py"
echo "│   ├── models/"
echo "│   │   ├── __init__.py"
echo "│   │   └── vae.py"
echo "│   ├── training/"
echo "│   │   ├── __init__.py"
echo "│   │   └── trainer.py"
echo "│   ├── inference/"
echo "│   │   ├── __init__.py"
echo "│   │   └── generator.py"
echo "│   └── utils/"
echo "│       ├── __init__.py"
echo "│       └── visualization.py"
echo "└── tests/"
echo "    ├── __init__.py"
echo "    └── test_data_loader.py" 