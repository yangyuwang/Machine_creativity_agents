# Machine Creativity Agents

Final project for UChi MACS37005 AI Agents for Social Science & Society

Author: Yangyu Wang, Qiuyi Yu, Yanjing Li, Yanran Qiu

Date: March 13, 2026

## Overview

This project explores machine creativity through latent diffusion models for historical artwork generation and simulation-based artistic evolution.

The project consists of three main components:

1. **Steering Module**: Mechanistic interpretability research using causal tracing between clean and creative prompts on a latent diffusion model trained on historical European paintings.

2. **Reinforcement Learning Module**: Reinforcement learning (RL) in improving machine creativity by optimizing creativity directly as an external objective.

3. **Simulation Module**: An agent-based model that simulates artistic evolution and patronage dynamics.

## Folder Structure

**Code/**

Contains all the code used in this project, organized into three main folders:

1. **Code/1steering/**

   This folder contains the notebook used to train the diffusion model, analyze internal representations, identify important neurons, and demonstrate interactive steering of image attributes.
   
3. **Code/2RL/**

   This folder contains notebooks used to construct style embeddings, design RL rewards, train reinforcement learning models, and evaluate performance.

3. **Code/3simulation/**

   This folder contains python files, configuration files and outputs of the exposure -> learning -> creation -> redistribution -> exposure simulation loop for agent creation.


