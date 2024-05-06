python test.py --mode VGG --ckpt_path vgg_trained.ckpt --test_bsz 1 --save_name 'normal_vgg.png'

python test.py --mode VGG --ckpt_path vgg_trained.ckpt --test_bsz 1 --save_name 'noise_vgg.png' --attack Noise

python test.py --mode VGG --ckpt_path vgg_trained.ckpt --test_bsz 1 --save_name 'fgsm_vgg.png' --attack FGSM

python test.py --mode VGG --ckpt_path vgg_trained.ckpt --test_bsz 1 --save_name 'pgd_vgg.png' --attack PGD

python test.py --mode VGG --ckpt_path vgg_trained.ckpt --test_bsz 1 --save_name 'cw_vgg.png' --attack CW

python test.py --mode VGG --ckpt_path vgg_trained.ckpt --test_bsz 1 --save_name 'pgd_targeted_vgg.png' --attack PGD --targeted --label 3,8