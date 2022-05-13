This is our solution used for IEEE access paper(https://ieeexplore.ieee.org/document/9585109).
Introduction: 

Our proposed MH UNet is a more profound, flexible, and lightweight architecture for medical image segmentation.
MH UNet has offered the following contributions:
1. We develop a novel multiscale hierarchical architecture for medical image segmentation. Dense connections allow deep supervision, smooth gradients flow, and reduced learnable parameters. Meanwhile, the residual-inception blocks extract multiscale features for robust representation.
2. The hierarchical block efficiently combines the multiscale local and global contexts in an encoder-decoder architecture. The hierarchical block improves the receptive field sizes of the dense blocks’ feature maps by different parallel dilation rates at the encoder of 3D UNet.
3. We present a deep supervision approach for faster convergence and superior segmentation accuracy. All dense blocks generate multiscale segmentation maps in the decoder. These multiscale segmentation maps are aggregated to boost the model’s convergence speed and accuracy.
4. We propose a combination of binary cross-entropy and dice loss functions to deal with severe class imbalance problems. Our model achieves significant segmentation accuracy due to the combined loss function, which does not require sophisticated weight hyper-parameter tuning.
5. We propose an efficient and simple post-processing technique to eliminate false-positives voxels.
6. We have used MICCAI BraTS and ISLES datasets for experimentation. Our proposed model outperformed all other state-of-the-art methods, including cascaded and ensembled approaches.

Development Environment:
Both my desktop and laptop had contributed a lot to the project.

Server:

V100 GPU of 32 GB

ubuntu16.04 + virtualenv + python==3.6.2 + tensorflow-gpu==1.15.0 + keras==2.2.4


Packages:
Here are some packages you may need to install.

1. For n4itk bias correction (https://ieeexplore.ieee.org/document/5445030) preprocessing, you may need to install ants.

a.) Just follow the installation guide on their homepage here (
Development Environment:
Both my desktop and laptop had contributed a lot to the project.

Server:

V100 GPU of 32 GB

ubuntu16.04 + virtualenv + python==3.6.2 + tensorflow-gpu==1.15.0 + keras==2.2.4


Packages:
Here are some packages you may need to install.

1. For n4itk bias correction (https://ieeexplore.ieee.org/document/5445030) preprocessing, you may need to install ants.

a.) Just follow the installation guide on their homepage here (http://neuro.debian.net/install_pkg.html?p=ants)
b.) Add ants to your environment variable PATH, for instance like $ export PATH=${PATH}:/usr/lib/ants/

2. sudo apt-get install libhdf5-serial-dev 

3. pip install numpy, nibable, SimpleITK, tqdm, xlrd, pandas, progressbar, matplotlib, nilearn, sklearn, tables

4. For Instance Normalization, you may need to download and install keras-contrib
a.) git clone https://www.github.com/farizrahman4u/keras-contrib.git
b.) pip install <where you saved it>

You can also install instance normalization

pip install git+https://www.github.com/keras-team/keras-contrib.git


Please follow the "How to run it" section from here (https://github.com/woodywff/brats_2019). 

Acknowledgement:

Again, this work refers to Isensee et.al's paper (https://www.nature.com/articles/s41592-020-01008-z), ellisdg's repository (https://github.com/ellisdg/3DUnetCNN/tree/master/legacy) and brats_2019's repository (https://github.com/woodywff/brats_2019), and tureckova's repository (https://github.com/tureckova/ISLES2018). We deeply appreciate their contributions to the community. Many thanks to the host of the BraTS and ISLES datasets.
