face_hallucination/
├── config.py          ← all constants (paths, hyperparams, blur conditions)
├── utils.py           ← seed, crop, splits, normalisation, metrics, checkpoints
├── models.py          ← BasicCNN, BiChannelCNN, WarmupCosineScheduler, loaders
├── dataset.py         ← TrainValFaceDataset, FixedTestFaceDataset
├── trainer.py         ← train_model(), run_test_pipeline(), evaluate_*()
├── classical.py       ← SC1Solver, SC2Solver, SFHSolver, run_classical_pipeline()
├── main.py            ← unified argparse entry point
├── train_basic.sh     ← SLURM: train BasicCNN
├── train_bichannel.sh ← SLURM: train BiChannelCNN
├── test_models.sh     ← SLURM: test either model
└── run_classical.sh   ← SLURM: run all classical methods

Steps: 
#0. Load celeba image dataset into your workdir, the images should be in: 
<cluster workdir>/celeba/img_align_celeba
Please download the images from this google drive their img folder: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg 

# 1. Copy all files to your cluster workdir
scp -r face_hallucination/ tans0444@cluster:/home/msds/tans0444/

# 2. Update the variables for the following files
├── train_basic.sh
    ├── WORKDIR: set to cluster workdir
    ├── IMAGE_ROOT: set to the folder containing the CelebA dataset, <cluster workdir>/celeba/img_align_celeba
    ├── CKPT_DIR: set to the folder containing the basiccnn checkpoints, <cluster workdir>/results_basiccnn/checkpoints
├── train_bichannel.sh
    ├── WORKDIR: set to cluster workdir
    ├── IMAGE_ROOT: set to the folder containing the CelebA dataset, <cluster workdir>/celeba/img_align_celeba
    ├── BASIC_CKPT: set to the file containing the basiccnn model, <cluster workdir>/results_basiccnn/best_model_basiccnn.pth
    ├── BI_CKPT_DIR: set to the folder containing the bichannel checkpoints, <cluster workdir>/results_bichannel/checkpoints
├── run_classical.sh
    ├── WORKDIR: set to cluster workdir
    ├── set to the folder containing the CelebA dataset, <cluster workdir>/celeba/img_align_celeba
├── test_models.sh
    ├── WORKDIR: set to cluster workdir
    ├── IMAGE_ROOT: set to the folder containing the CelebA dataset, <cluster workdir>/celeba/img_align_celeba
    ├── DEFAULT_CKPT (for test_basic): set to <cluster workdir>/results_basiccnn/best_model_basiccnn.pth
    ├── DEFAULT_CKPT (for test_bichannel): <cluster workdir>/results_bichannel/best_model.pth
For experiment parameters such as epochs or learning rate, please adjust values found in config.py

#3. Prepare DL2 environment, run the following lines in the terminal
chmod 777 setup/setup.sh
./setup.sh
If the setup script could not be ran, please install the dependencies defined in setup/environment.yaml for the DL2 conda environment

# 4. Train BasicCNN (~48 hrs)
sbatch train_basic.sh

# 5. Train BiChannelCNN (after BasicCNN finishes, ~48 hrs)
sbatch train_bichannel.sh

# 6. Test either model (~10 min)
sbatch test_models.sh basic
sbatch test_models.sh bichannel

# 7. Run classical methods (~15min, CPU only)
sbatch run_classical.sh

# 8. Or run anything directly without SLURM
python main.py --mode train_basic --image-root /path/to/img_align_celeba
python main.py --mode test_bichannel
python main.py --mode classical