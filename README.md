# Pytorch MIRNet Implementation (unofficial code)

## [[Paper]](https://arxiv.org/abs/2003.06792)
## Learning Enriched Features for Real Image Restoration and Enhancement (ECCV, 2020)


***Abstract***


*With the goal of recovering high-quality image content from its degraded version, image restoration enjoys numerous applications, such as in surveillance, computational photography, medical imaging, and remote sensing. Recently, convolutional neural networks (CNNs) have achieved dramatic improvements over conventional approaches for image restoration task. Existing CNN-based methods typically operate either on full-resolution or on progressively low-resolution representations. In the former case, spatially precise but contextually less robust results are achieved, while in the latter case, semantically reliable but spatially less accurate outputs are generated. In this paper, we present a novel architecture with the collective goals of maintaining spatially-precise high-resolution representations through the entire network and receiving strong contextual information from the low-resolution representations. The core of our approach is a multi-scale residual block containing several key elements: (a) parallel multi-resolution convolution streams for extracting multi-scale features, (b) information exchange across the multi-resolution streams, (c) spatial and channel attention mechanisms for capturing contextual information, and (d) attention based multi-scale feature aggregation. In a nutshell, our approach learns an enriched set of features that combines contextual information from multiple scales, while simultaneously preserving the high-resolution spatial details. Extensive experiments on five real image benchmark datasets demonstrate that our method, named as MIRNet, achieves state-of-the-art results for a variety of image processing tasks, including image denoising, super-resolution, and image enhancement. The source code and pre-trained models are available at https://github.com/swz30/MIRNet.*


![image](https://user-images.githubusercontent.com/33386742/153009287-0593ead8-ff71-4810-9029-670cff0a573b.png)

<br/>


## Train
```
> python train.py --epochs 100 --batch_size 16
```
<br/>



## Reference
* [Official code](https://github.com/swz30/MIRNet)
