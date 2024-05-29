# Blind-Face-Super-Resolution

The blind-face super resolution challenge requires creating high-quality (HQ) face images from low-quality (LQ) ones using a dataset from FFHQ. The dataset contains 5000 HQ training images and 400 LQ-HQ image pairs for validation. During training, LQ images are generated from HQ images using the random second-order degradation pipeline that includes Gaussian blur, downsampling, noise, and compression.

The implementation of Real-ESRGAN and EDSR are in directory call esrgan and EDSR.

For more details, check the report [here](https://drive.google.com/file/d/1tq3CDpnTUB1QtywEfqexZBeHmdCxJkn-/view?usp=sharing).

## Real-ESRGAN

For this project, please put the FFHQ dataset file inside the realesrgan\data directory. In which FFHQ folder immediately contains train, val, test directory.

The implementation of Real-ESRGAN is in the esrgan folder. Please excecute ESRGAN command inside esrgan directory.

To reproduce the result:
first run: 
python realesrgan/train.py -opt options/train_SRResNet_x4_FFHQ_300k.yml --auto_resume

then run:
python realesrgan/train.py -opt options/train_SRResNet_x4_FFHQ_300k_cont.yml --auto_resume


To generate best images, run the following line in esrgan directory:
python infer.py


To run the heavier degradation training:
run:

python realesrgan/train.py -opt options/train_SRResNet_x4_FFHQ_300k_40kep.yml --auto_resume


## EDSR

For this project, please put the FFHQ dataset file inside the data directory at the root folder. In which FFHQ folder immediately contains train, val, test directory.

To reproduce result of EDSR, run this line in direcory call 'EDSR-PyTorch-edited', it's in EDSR dir:
python src\\main.py --model EDSR --scale 4 --save edsr_no_meanshift --patch_size 64 --reset --epochs 100 --save_result

Generate images, run this line in direcory call 'EDSR-PyTorch-edited', it's in EDSR dir:
python src\\main.py --model EDSR --scale 4 --patch_size 96 --reset --epochs 100 --resume 0 --save_result --test_only --pre_train ..\\experiment\\edsr_no_meanshift\\model\\model_best.pt --save_gt


HiFaceGAN:

Too large not applicable.


Reference:
Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
EDSR: https://github.com/sanghyun-son/EDSR-PyTorch/tree/master?tab=readme-ov-file
HiFaceGAN: https://github.com/Lotayou/Face-Renovation