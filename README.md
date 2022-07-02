# Use W-Net to jointly do self-supervised depth estimation and unsupervised segmentation  
W-Net: A Deep Model for Fully Unsupervised Image Segmentation https://arxiv.org/pdf/1711.08506.pdf  
Digging into Self-Supervised Monocular Depth Prediction https://arxiv.org/abs/1806.01260  

### Architecture
RGB Image ---> U-Net ---> Segmentation ---> U-Net ---> Depth Map  
![1656746355(1)](https://user-images.githubusercontent.com/42352462/176990885-56cf841e-8ab1-499b-a2f8-16b926e06236.png)

### Evaluation on KITTI  
mIoU 14.53  
![1656745816(1)](https://user-images.githubusercontent.com/42352462/176990623-e7de3731-3587-4b34-a79e-6efa5c0bc624.png)  




Code is based on Monodepth2

> **Digging into Self-Supervised Monocular Depth Prediction**
>
> [Clément Godard](http://www0.cs.ucl.ac.uk/staff/C.Godard/), [Oisin Mac Aodha](http://vision.caltech.edu/~macaodha/), [Michael Firman](http://www.michaelfirman.co.uk) and [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/)  
>
> [ICCV 2019 (arXiv pdf)](https://arxiv.org/abs/1806.01260)

