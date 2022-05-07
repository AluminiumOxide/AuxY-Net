### Since it is not very specification |  this project is for reference only for the time being
##### I propose a multi-head semantic segmentation network structure called AuxY-Net based on U-Net
##### his structure allows to achieve better performance in quantitative photoacoustic imaging with more robust 
- \# The detailed comparison information will be updated here after the madness, suffocation, nausea, and despair CNKI CHECK

---
An intermediate network structure called AuxU_Net built as prototype in the middle,
to avoid repetition, it is called FLAuxU-Net in my Graduation Project

Now This is the structure about this project

>  bins  # useless or proto tools when I working
> 
> data  # add costume dataset here
> 
> datasets  # which dir store dataset, I left a few simulated brain dataset mats and catalog format for reference  
>
> > brain  # example dataset
> > > train_fai  # example mul train set dir
> > > > - *.mat  # example mat image
> > > >
> > > > - *.mat
> > > >
> > > > - ......
> > > 
> > > train_p0
> > > 
> > > train_p1
> > >
> > > train_ua
> > >
> > > val_fai # example mul val set dir
> > > 
> > > val_p0
> > > 
> > > val_p1
> > >
> > > val_ua
> > >
> 
> nets # add costume net and all model store at here
>
> workdir #  Store network results and parameters
> 
> - step1_*  # traditional model(PSPNet、DeepLab、UNet、R2UNet、AttnUNet、UNet++) train code  
> 
> - step2_*  # Update model train code and U-Net with the same parameters set
> 
> - step3_*  # Update model eval code
> 
> - tool1_net_profile # Show network profile to a csv file

---

Now I share a spare space for those model structure, Ill add it when I will add it after the graduation defense .


However, some parameters can still be displayed in advance,such as network profile

| index |   model    |      macs      |   params   |
|:-----:|:----------:|:--------------:|:----------:|
|   0   |   U_Net    | 65475379200.0  | 34527171.0 |
|   1   |  R2U_Net   | 152824119296.0 | 39091523.0 |
|   2   |  AttU_Net  | 66576907264.0  | 34878703.0 |
|   3   | R2AttU_Net | 153925647360.0 | 39443055.0 |
|   4   | NestedUNet | 138601496576.0 | 36629763.0 |
|   5   |  AuxU_Net  | 112994418688.0 | 49782546.0 |
|   6   |  AuxY_Net  | 95940507648.0  | 35856938.0 |
|   7   |   PSPNet   | 46149837824.0  | 46707139.0 |
|   8   | DeepLabV3  | 40961578496.0  | 39634243.0 |

I do not provide public datasets for now
If you want to get in touch further, please send an email to [Inferno5415@outlook.com](inferno5415@outlook.com)
Or visit my [personal website](http://allophane.com/) for reference.
