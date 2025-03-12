CUDA_VISIBLE_DEVICES='0' python inference_ofer.py \
   --input_folder  /mnt/d/Projects/Deep_Learning/final_project/databases/voicebank-demand/noisy_testset_wav \
   --output_folder results/fusion/testset \
   --checkpoint_file exp_fusion/SEMamba_fusion_v1/g_00530000.pth  \
   --config recipes/SEMamba_fusion/SEMamba_fusion.yaml \
   --post_processing_PCS False \
   
