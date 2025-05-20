FPGA-based EEG Denoising with Deep Autoencoder
This repository contains the complete workflow for implementing and evaluating a Deep Autoencoder (DAE) model for EEG signal denoising on FPGA. The project is divided into the following main components:

Repository Structure
Root Directory
modified DAE/: Optimized DAE model adapted for FPGA deployment (e.g., layer structure adjusted, quantized version).
original_DAE/: Original model used for reference and baseline performance comparison.
power measurement/: Scripts, logs, and documentation related to runtime power profiling of the FPGA implementation.

Subdirectory (within modified DAE)
1_model_construction/: MATLAB/Python code for building and converting the DAE model structure.
2_deploy_on_FPGA/: HDL generation, Vivado project files, and deployment scripts for the Zynq-based FPGA board.
3_result_analyse/: Post-deployment analysis including denoising quality (RRMSE, CC) and performance metrics (latency, energy).
