# AFormer - a Transformer Based Model for Pan-sharpening

This repository contains the code implementation for our Transformer based model "AFormer" for Pan-sharpening

## Requirements
- Python 3.7+
- TensorFlow 2.4+
- NumPy
- OpenCV
- Matplotlib
- Scikit-image
- tqdm
- scikit-learn

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/nitingargg/AFormer-a-Transformer-Based-Model-for-Pan-sharpening.git

    cd AFormer-a-Transformer-Based-Model-for-Pan-sharpening
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
## Usage
1. Prepare your dataset in the required format.
2. Modify the configuration file `config.yaml` to set your parameters.
3. Train the model:
    ```bash
    python train.py --config config.yaml
    ```
4. Evaluate the model:
    ```bash
    python evaluate.py --config config.yaml
    ```
5. Test the model:
    ```bash
    python test.py --config config.yaml
    ```
## Results
After training and testing, the results will be saved in the `results/` directory. You can visualize the pan-sharpened images and compare them with the ground truth. 

## Thank you!
