# Male-Female Face Generator (DCGAN)

---
## Tech Stack

---
## Introduction

---
## Implementation

---
This is the **test** branch for this repo.
The following steps are to be followed to run this repo:
1. Run the bash script.
    ```bash
    $ ./bashScript.sh
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

Training began from noise like this:

![Image](Generated_images/Generated_Images_at_Epoch_0000.png)

... to more satisfactory results.


Said results are displayed below:

---

Epoch 1:

![Image](Generated_images/Generated_Images_at_Epoch_0001.png)
---
Epoch 10:

![Image](Generated_images/Generated_Images_at_Epoch_0010.png)
---
Epoch 20:

![Image](Generated_images/Generated_Images_at_Epoch_0020.png)
---
Epoch 30:

![Image](Generated_images/Generated_Images_at_Epoch_0030.png)
---
Epoch 40:

![Image](Generated_images/Generated_Images_at_Epoch_0040.png)
---
Epoch 50:

![Image](Generated_images/Generated_Images_at_Epoch_0050.png)
---

All results obtained above were obtained via training with the code snippet below:
```bash
cd scripts
python main.py --epochs 50 --lr 2e-4 --decay_rate .001 --style solarizedd
```
## To-Dos

---
There are still a few additions to make to the project. They include:
1. Improve documentation.
2. Implement training via **noisy labels**.
3. Try out other GAN variants.

