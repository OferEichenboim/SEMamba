CUDA_VISIBLE_DEVICES='0' python inference_fusion.py \
   --input_folder  /mnt/d/Projects/Deep_Learning/final_project/databases/voicebank-demand/noisy_testset_wav \
   --output_folder results/fusion/testset \
   --checkpoint_file ckpts/Semamba_fusion_00138000_PESQ_3_291.pth  \
   --config recipes/SEMamba_fusion/SEMamba_fusion.yaml \
   --post_processing_PCS False \
   
