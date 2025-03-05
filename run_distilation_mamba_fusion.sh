CUDA_VISIBLE_DEVICES='0' python train_semamba_distilation.py \
  --config recipes/SEMamba_distilation/SEMamba_distilation_fusion.yaml \
  --exp_folder exp_fusion_distilation \
  --exp_name SEMamba_fusion_distilation_v1 \
  --checkpoint_file ckpts/SEMamba_advanced.pth  \
