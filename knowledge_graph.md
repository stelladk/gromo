# Gromo Knowledge Graph

## Core Components

### Base Classes
- **GrowingModule** (`src/gromo/modules/growing_module.py`)
  - Base class for all growable modules
  - Provides core functionality for network growth

- **GrowingContainer** (`src/gromo/containers/growing_container.py`)
  - Base class for containers holding growable modules
  - Manages collections of growable modules

### Module Implementations
- **LinearGrowingModule** (`src/gromo/modules/linear_growing_module.py`)
  - Implements growable linear (dense) layers
  - Extends GrowingModule

- **Conv2dGrowingModule** (`src/gromo/modules/conv2d_growing_module.py`)
  - Implements growable 2D convolutional layers
  - Extends GrowingModule

- **GrowingNormalization** (`src/gromo/modules/growing_normalization.py`)
  - Implements normalization layers
  - Supports various normalization techniques

### Container Implementations
- **GrowingDAG** (`src/gromo/containers/growing_dag.py`)
  - Implements directed acyclic graphs of growable modules
  - Manages complex network topologies

- **GrowingGraphNetwork** (`src/gromo/containers/growing_graph_network.py`)
  - Implements graph neural networks with growable components
  - Supports graph-based operations

- **GrowingMLP** (`src/gromo/containers/growing_mlp.py`)
  - Implements multi-layer perceptron with growable layers
  - Simplifies creation of MLP networks

- **GrowingMLPMixer** (`src/gromo/containers/growing_mlp_mixer.py`)
  - Implements MLP-Mixer architecture with growable components

## Dependencies
- **Core Dependencies**:
  - PyTorch (torch)
  - NumPy
  - NetworkX

## Project Structure
```
gromo/
├── src/
│   └── gromo/
│       ├── config/         # Configuration management
│       ├── containers/     # Container implementations
│       │   ├── growing_block.py
│       │   ├── growing_container.py
│       │   ├── growing_dag.py
│       │   ├── growing_graph_network.py
│       │   ├── growing_mlp.py
│       │   ├── growing_mlp_mixer.py
│       │   └── growing_residual_mlp.py
│       ├── modules/        # Module implementations
│       │   ├── attention/
│       │   ├── constant_module.py
│       │   ├── conv2d_growing_module.py
│       │   ├── growing_module.py
│       │   ├── growing_normalisation.py
│       │   └── linear_growing_module.py
│       └── utils/          # Utility functions
└── tests/                  # Test suite
```

## Relationships
- All growable modules inherit from `GrowingModule`
- All container classes inherit from `GrowingContainer`
- The system uses PyTorch's autograd for automatic differentiation
- NetworkX is used for graph operations in `GrowingDAG` and `GrowingGraphNetwork`

## Usage Patterns
1. Create growable modules (e.g., `LinearGrowingModule`, `Conv2dGrowingModule`)
2. Combine them using containers (e.g., `GrowingMLP`, `GrowingDAG`)
3. Train the network using standard PyTorch training loops
4. Use built-in growth mechanisms to expand the network as needed
