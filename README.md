# Male-Female Face Generator (DCGAN)

---
## Tech Stack

---
## Introduction

---
## Implementation

---
The following steps are to be followed to run this repo:
1. Run the bash script.
    ```bash
    $ sh bashScript.sh
    ```
   This will download the image dataset and unzip it into the `dataset` directory.
2. Run the `main.py` script. This takes a number of arguments including:
    ```
    - figsize: image size for generated images (Default: 5 X 5)
    - lr: speed of convergence (Default: 1e-4)
    - data_dir: Dataset directory
    - image_dir: Directory to save generated images
    - stats: image normalization statistics
    ```
   ...amongst others.
3. 
---
## Results

Training began from noise:

... to satisfactory results.
The results from training may be viewed below:

Epoch 1:

![Image](Generated images/Generated Images at Epoch 0001.png)
---
Epoch 10:

![Image](Generated images/Generated Images at Epoch 0010.png)
---
Epoch 20:

![Image](Generated images/Generated Images at Epoch 0020.png)
---
Epoch 30:

![Image](Generated images/Generated Images at Epoch 0030.png)
---
Epoch 40:

![Image](Generated images/Generated Images at Epoch 0040.png)
---
Epoch 50:

![Image](Generated images/Generated Images at Epoch 0050.png)
---
## To-Dos

---
There are still a few additions to make to the project. They include:
1. Implement training via **noisy labels**.
2. 

