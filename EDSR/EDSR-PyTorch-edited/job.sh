python src\\main.py --model EDSR --scale 4 --save edsr_baseline_x4 --patch_size 96 --reset --epochs 50

python src\\main.py --model EDSR --scale 4 --save edsr_baseline_x4 --patch_size 96 --reset --epochs 100 --resume best --save_result --test_only

python src\\main.py --model EDSR --scale 4 --save edsr_baseline_x4 --patch_size 96 --reset --epochs 100 --resume 0 --save_result --test_only --save_gt --pre_train ..\\experiment\\edsr_baseline_x4\\model\\38_model_best.pt

100 epoch:
python src\\main.py --model EDSR --scale 4 --save edsr_no_meanshift --patch_size 64 --reset --epochs 100 --save_result

test:
python src\\main.py --model EDSR --scale 4 --patch_size 96 --reset --epochs 100 --resume 0 --save_result --test_only --pre_train ..\\experiment\\edsr_no_meanshift\\model\\model_best.pt --save_gt

python src\\main.py --model MDSR --scale 4 --save edsr_with_shift --patch_size 64 --reset --epochs 100 --save_result