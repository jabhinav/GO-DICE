# Installation Instructions

## Compatibility

- Compatible with TensorFlow 2.x
- Compatible with arm64 /x86_64 architecture

## TensorFlow Installation on M1/M2 Macs with native GPU support

### Step 1: Install conda

Download the conda installer from [here](https://www.anaconda.com/download). Make sure to download the arch. specific conda can from [here](https://www.anaconda.com/download).

### Step 2: Create a conda environment

```bash
conda create -n tf-gpu python=3.8
```

### Step 3: Activate the conda environment

```bash
conda activate tf-gpu
```

### Step 4: Install TensorFlow

```bash
pip install tensorflow && pip install tensorflow-metal
```

Note: tensorflow-metal is required for GPU support on M1/M2 Macs. Details [here](https://developer.apple.com/metal/tensorflow-plugin/).

### Step 5: Verify the installation

```bash
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

## Install Required Packages

```bash
pip install -r requirements.txt
```

Note: The `latest` branch uses `gymansium` and `mujoco` for the environments. The `dev` branch uses `gym` and `mujoco-py` for the environments.

## Resources

- [Gymnasium](https://gymnasium.farama.org/content/basic_usage): Gymnasium (maintained by Farama) is the now-supported version of OpenAI Gym.
- [Gymnasium-Robotics](https://robotics.farama.org): Robotics environments for Gymnasium provided separately by Farama.
- [Mujoco](https://mujoco.readthedocs.io/en/stable/python.html): MuJoCo is a physics engine for detailed, efficient rigid body simulations with contacts. Starting with version 2.1.2, MuJoCo comes with native Python bindings. Earlier they needed to be installed separately via `mujoco-py`.