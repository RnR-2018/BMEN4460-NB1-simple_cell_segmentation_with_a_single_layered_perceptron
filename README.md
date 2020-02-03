# BMEN4460-Notebook1
## Simple Cell Segmentation with a Single Layered Perceptron.
Nanyan "Rosalie" Zhu and Chen "Raphael" Liu

This repository is a child repository of [**RnR-2018/Deep-learning-with-PyTorch-and-GCP**](https://github.com/RnR-2018/Deep-learning-with-PyTorch-and-GCP). This serves a primary purpose of facilitating the course BMEN4460 instructed by Dr. Andrew Laine and Dr. Jia Guo at Columbia University. However, it can also be used as a general beginner-level tutorial to implementing deep learning algorithms with PyTorch on Google Cloud Platform.

This repository, [**Simple Cell Segmentation with a Single Layered Perceptron**](https://github.com/RnR-2018/BMEN4460-NB1-simple_cell_segmentation_with_a_single_layered_perceptron), presents three simple examples of the least complicated neural networks. For students in BMEN4460 (or who follow the instructions Step00 through Step02 in the parent repository), please create a Projects folder within your GCP VM and download this repository into that folder.

On the GCP VM Terminal:
```
cd /home/[username]/
mkdir Projects
cd Projects
git clone https://github.com/RnR-2018/BMEN4460-NB1-simple_cell_segmentation_with_a_single_layered_perceptron/
```

You shall then see the following hierarchy of files and folders, hopefully, which matches the hierarchy of the current repository.

```
BMEN4460-NB1-simple_cell_segmentation_with_a_single_layered_perceptron
    ├── BMEN4460_NB1_simple_cell_segmentation_with_a_single_layered_perceptron.ipynb
    ├── helper.py
    └── data
        └── sample_cell_image.tif
        └── perceptron_single_input.PNG
        └── perceptron_multi_input.PNG
        └── perceptron_CNN.PNG
```

Again, the concepts covered are quite rudimentary. Hope you enjoy this.

## Figures we created to illustrate the three perceptron candidates.
### Candidate 1: Perceptron, single input.
<img src="/data/perceptron_single_input.PNG" width="1000px">

### Candidate 2: Perceptron, multi input.
<img src="/data/perceptron_multi_input.PNG" width="1000px">

### Candidate 3: Perceptron, convolutional neural network.
<img src="/data/perceptron_CNN.PNG" width="1000px">
