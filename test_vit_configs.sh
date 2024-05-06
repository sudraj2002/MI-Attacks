CUDA_VISIBLE_DEVICES=1 python test.py --mode ViT --ckpt_path vit_trained.ckpt --test_bsz 1 --save_name 'normal_vit.png'

CUDA_VISIBLE_DEVICES=1 python test.py --mode ViT --ckpt_path vit_trained.ckpt --test_bsz 1 --save_name 'noise_vit.png' --attack Noise

CUDA_VISIBLE_DEVICES=1 python test.py --mode ViT --ckpt_path vit_trained.ckpt --test_bsz 1 --save_name 'fgsm_vit.png' --attack FGSM

CUDA_VISIBLE_DEVICES=1 python test.py --mode ViT --ckpt_path vit_trained.ckpt --test_bsz 1 --save_name 'pgd_vit.png' --attack PGD

CUDA_VISIBLE_DEVICES=1 python test.py --mode ViT --ckpt_path vit_trained.ckpt --test_bsz 1 --save_name 'cw_vit.png' --attack CW

CUDA_VISIBLE_DEVICES=1 python test.py --mode ViT --ckpt_path vit_trained.ckpt --test_bsz 1 --save_name 'pgd_targeted_vit.png' --attack PGD --targeted --label 3,8