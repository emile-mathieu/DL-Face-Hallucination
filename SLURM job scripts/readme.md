4 main job scripts
1. Train Bi-channel CNN
2. Train Basic CNN
3. Run eval_baselines once, passing both checkpoints
- This trains SC1 and SC2
- Builds SFH from SC1
- Validates Bicubic/SC1/SC2/SFH
- Tests all 4 methods (Bicubic, SC1, SC2, SFH)
- Prints PSNR/SSIM for Gaussian σ=1,3,5 and motion l=2,6,9
4. Run eval_motion and eval_Gaussian on to test bichannel CNN and basic CNN 
- Prints PSNR/SSIM for Gaussian σ=1,3,5 and motion l=2,6,9

2 side job script: 
1. Print table (original paper table 2) 
2. Obtain sample image after preprocessing is done 