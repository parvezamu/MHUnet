Codes Coming Soon

This is our solution used for IEEE access paper(https://ieeexplore.ieee.org/document/9585109). 

Our proposed MH UNet is a more profound, flexible, and lightweight architecture for medical image segmentation.
MH UNet has offered the following contributions:
1. We develop a novel multiscale hierarchical architecture for medical image segmentation. Dense connections allow deep supervision, smooth gradients flow, and reduced learnable parameters. Meanwhile, the residual-inception blocks extract multiscale features for robust representation.
2. The hierarchical block efficiently combines the multiscale local and global contexts in an encoder-decoder architecture. The hierarchical block improves the receptive field sizes of the dense blocks’ feature maps by different parallel dilation rates at the encoder of 3D UNet.
3. We present a deep supervision approach for faster convergence and superior segmentation accuracy. All dense blocks generate multiscale segmentation maps in the decoder. These multiscale segmentation maps are aggregated to boost the model’s convergence speed and accuracy.
4. We propose a combination of binary cross-entropy and dice loss functions to deal with severe class imbalance problems. Our model achieves significant segmentation accuracy due to the combined loss function, which does not require sophisticated weight hyper-parameter tuning.
5. We propose an efficient and simple post-processing technique to eliminate false-positives voxels.
6. We have used MICCAI BraTS and ISLES datasets for experimentation. Our proposed model outperformed all other state-of-the-art methods, including cascaded and ensembled approaches.

Many thanks to the host of the BraTS and ISLES datasets.
