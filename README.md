# S.L.A.V.E: **S**elf **L**earning **A**gent for **V**ariable **E**nvironments

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge)

An open-source exploration into pure, vision-based Reinforcement Learning. This project implements a simulated 4-DOF (Degrees of Freedom) robotic arm that learns to interact with its environment using absolutely no hardcoded kinematic paths, no audio, and no language processing. It relies solely on a custom Convolutional Neural Network (CNN) processing raw pixel data from a claw-mounted camera.

The agent "just sees, experiences, and learns."

# Creator: Satvik Sengupta (https://github.com/SatvikSengupta)

## Architecture Overview

To solve the sample-inefficiency problem of Reinforcement Learning from Human Feedback (RLHF) on a vision-based model, this project utilizes a two-phase training pipeline bridging a custom PyTorch brain with a PyBullet physics simulation.

### The Brain (`model.py` & `train.py`)
*   **Vision Encoder:** A lightweight CNN that compresses 128x128 RGB camera frames into a 512-dimensional latent spatial vector.
*   **Advantage Actor-Critic (A2C):** A dual-headed policy network. The Actor maps the latent vector to 4 continuous joint target angles, while the Critic estimates the value of the current visual state to stabilize gradient updates.
*   **Frame Skipping:** Physics steps are looped at 10x the decision rate to prevent action-chattering and allow the virtual servos to execute commands smoothly.

### The Body & Environment (`env.py` & `arm.urdf`)
*   **Simulated Hardware:** A mathematically proportioned arm built in URDF with a 2:1.5:1 segment ratio. It consists of a Yaw shoulder joint, and three Pitch joints (elbow, wrist, neck) for high-articulation reaching.
*   **Gymnasium Wrapper:** The PyBullet physics engine is wrapped in a standard `gym.Env` interface to handle asynchronous physics stepping, synthetic camera rendering, and reward calculation.

### The Interface (`feedback_ui.py` & `master.py`)
*   **Inter-Process Communication (IPC):** Human feedback is sparse. To prevent blocking the 240Hz physics thread, the Tkinter UI runs on a separate process and broadcasts `+1`, `0`, or `-1` reward signals via a local UDP socket (Port 5005) directly to the environment.


## Version History

| Version | Major Changes |
|----------|---------------|
| v1.1 | Human Reward interface, PyBullet Environment, Main model |
| v1.2 | Automated reward architecture, human reward interface removed |
| v1.3 | Upgraded manipulator from 3 DOF → 4 DOF |
| v1.4 | Workspace expansion |

## Usage

### Install Required Dependencies:
``` bash
pip install torch pybullet gymnasium opencv-python numpy
```
##### Note:
In case PyBullet does not install, create a virtual environment (venv) with:
``` bash
pythom -m venv venv
```

### Run Master Script (v1.2 onwards):
``` bash
python master.py
```

## Project Structure:
* **master.py:** Subprocess manager that boots and synchronizes the training loop and UI.
* **arm.urdf:** The XML-based kinematic blueprint defining the physical masses, joints, and camera anchor of the arm.
* **env.py:** The Gymnasium environment handling PyBullet physics, camera rendering, and the UDP socket listener.
* **model.py:** The PyTorch architecture containing the CNN Vision Encoder and the A2C neural network heads.
* **train.py:** The reinforcement learning loop executing the Forward Pass, Action Sampling, and Backpropagation.
* **feedback_ui.py:** The standalone Tkinter application for broadcasting RLHF signals. (currently not in use)

## License
Under Apache 2.0, see **LICENSE** for more
