##Use W-Net to jointly do self-supervised depth estimation and unsupervised segmentation
RGB Image ---> U-Net ---> Segmentation ---> U-Net ---> Depth Map
![1656745378(1)](https://user-images.githubusercontent.com/42352462/176990441-c69a953a-fbf7-43e0-9e2c-98e54ee8528f.png)

#Evaluation on KITTI
mIoU 14.53
![1656745816(1)](https://user-images.githubusercontent.com/42352462/176990623-e7de3731-3587-4b34-a79e-6efa5c0bc624.png)




Code is based on Monodepth2

> **Digging into Self-Supervised Monocular Depth Prediction**
>
> [Clément Godard](http://www0.cs.ucl.ac.uk/staff/C.Godard/), [Oisin Mac Aodha](http://vision.caltech.edu/~macaodha/), [Michael Firman](http://www.michaelfirman.co.uk) and [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/)  
>
> [ICCV 2019 (arXiv pdf)](https://arxiv.org/abs/1806.01260)

