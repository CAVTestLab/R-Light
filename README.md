# R-Light

A reinforcement learning framework for network-level traffic signal control with PyTorch and SUMO simulation platform support.
In this project, the following files will be provided:
1. We supply the code for two methods (D3QN and GCN) used for traffic signal control.
2. The layout of Hangzhou and Jinan maps (reconstructed versions) as presented in the paper is included in the “env” folder.
3. A performance video of the proposed model, R-Light, is provided. The relevant code will be uploaded once the paper is published.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![SUMO](https://img.shields.io/badge/SUMO-1.8.0+-green.svg)](https://sumo.dlr.de/)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Performance Demo](#performance-demo)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

R-Light is a comprehensive reinforcement learning framework designed for intelligent traffic signal control at the network level. It provides state-of-the-art algorithms for optimizing traffic flow through adaptive signal timing, leveraging deep reinforcement learning techniques integrated with the SUMO (Simulation of Urban Mobility) platform.

## Features

### Core Algorithms
- **DQN-GCN**: Deep Q-Network with Graph Convolutional Networks for spatial traffic pattern learning
- **D3QN**: Dueling Double Deep Q-Network for enhanced value estimation
- **R-Light**: Our proposed algorithm with superior performance metrics

### Key Capabilities
- Real-time traffic signal optimization
- Network-level coordination across multiple intersections
- Integration with SUMO simulation environment
- Comprehensive training and evaluation framework
- Support for real-world traffic datasets

## Installation

### Prerequisites
- Python >= 3.8
- SUMO >= 1.8.0
- PyTorch >= 1.9.0

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/R-Light.git
cd R-Light
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure SUMO environment:
```bash
# Add SUMO to your PATH or set SUMO_HOME environment variable
export SUMO_HOME="/path/to/sumo"
```

## Quick Start

### Training
```bash
python train.py --config config.py --algorithm r-light
```

### Evaluation
```bash
python evaluate.py --model path/to/trained/model --scenario hangzhou_4x4
```

## Project Structure

```
R-Light/
├── LibSignal/          # Traffic signal control library
├── agent/              # RL agent implementations
├── core/               # Core functionality modules
├── env/                # Environment wrappers and scenarios
├── trainers/           # Training orchestration
├── video/              # Performance demonstration videos
├── config.py           # Configuration parameters
├── requirements.txt    # Python dependencies
├── train.py            # Main training script
├── evaluate.py         # Evaluation script
├── README.md           # This file
└── LICENSE             # License information
```

## Datasets

### Real-World Traffic Data

We provide validated real-world traffic datasets from major Chinese cities:

#### Hangzhou Dataset
- **Network**: 4×4 intersection grid
- **Scenarios**: 2 different traffic flow patterns
- **Location**: `env/Intersection/hangzhou/`

#### Jinan Dataset  
- **Network**: 3×4 intersection grid
- **Scenarios**: 3 different traffic flow patterns
- **Location**: `env/Intersection/jinan/`

## Performance Demo

Our repository includes comprehensive performance demonstrations:

- **Video Demonstrations**: Real-time metric comparisons between R-Light and FixedTime baseline
- **Performance Visualizations**: Detailed charts showing improvement across multiple KPIs
- **Location**: `video/` directory
> 📹 **[Click here to view the performance comparison video](video/comparison.mp4)**

### Key Performance Improvements
- Reduced average waiting time
- Improved traffic throughput
- Enhanced network-level coordination
- Superior adaptability to traffic pattern changes


## Contributing

We welcome contributions!

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

- SUMO development team for the simulation platform
- Research collaborators and data providers
- Open-source community contributions

---

For questions and support, please open an issue or contact [fang6100146@gmail.com](mailto:fang6100146@gmail.com).


