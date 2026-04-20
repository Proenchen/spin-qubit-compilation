# Spn Qubit Compilation

This repository contains the implementation of the routing algorithms and inital-mapping techniques described in the paper "_Compilation Techniques for Error Corrected Silicon Spin Qubit Quantum Computers_".

## Usage

Follow these steps to execute the framework:
1. Clone this repository using
   
   ```
   git clone https://github.com/Proenchen/spin-qubit-compilation.git
   ```
2. Install all required packages listed in the [requirements](requirements.txt) by running:
   
   ```
   pip install -r requirements.txt
   ```
3. In `main.py`, select the desired initial-mapping and routing strategy, and adjust the `SimulationConfig` if needed.
4. Run the program by executing `main.py`.

After that, a GUI should appear that visualizes the routing simulation:

<img width="659" height="556" alt="image" src="https://github.com/user-attachments/assets/7e75a36e-8c68-45e6-80ec-59c5cb7dae06" />

