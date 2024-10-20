# Deep Learning Project Template using in Cluster with SLURM and CUDA
This template is to train a deep learning project by SLURM for allocating single GPU and single server environment.


### Project Template
```
deep-learning-project/
│
├── config/
│   └── config.yaml            # Configuration file for the project (e.g., hyperparameters, paths, etc.)
│
├── data/
│   └── dataset/               # Folder where your datasets will reside
│
├── logs/
│   └── training.log           # Logs of training runs (stdout or custom logs)
│
├── models/
│   └── model_checkpoint.pth   # Checkpoint of your trained model
│
├── scripts/
│   └── train.py               # Script to run training
│   └── utils.py               # Helper functions for data preprocessing, logging, etc.
│
├── notebooks/
│   └── exploration.ipynb       # Jupyter notebooks for model exploration
│
├── requirements.txt            # Python dependencies
├── README.md                   # Overview of the project and how to run it
└── slurm_job.sh                # SLURM job script to submit job
```
### Model Configuration
you can change model parameters and hyperparameters,
located inside ```config/config.yaml```

A sample is here:
```yaml
batch_size: 64
learning_rate: 0.001
num_epochs: 10
data_path: "./data/dataset"
log_path: "./logs/training.log"
model_save_path: "./models/model_checkpoint.pth"

```

### Running the Training Script with SLURM

To run this script on your single GPU using SLURM, create a SLURM job script (included in template):

```bash
#!/bin/bash
#SBATCH --job-name=deep_learning_training
#SBATCH --output=logs/slurm-%j.out   # Save output logs to file
#SBATCH --error=logs/slurm-%j.err    # Save error logs to file
#SBATCH --partition=gpu              # Specify GPU partition (customize based on your setup)
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --ntasks=1                   # Request 1 task (process)
#SBATCH --mem=16G                    # Memory allocation for the job
#SBATCH --time=02:00:00              # Time limit for the job (example: 2 hours)

# Load modules if necessary (e.g., CUDA)
module load cuda/11.7

# Run the training script
python scripts/train.py
```

### How to Submit SLURM Job
To submit the SLURM job:
```bash
sbatch slurm_job.sh
```

### Verify job is submitted

```bash
squeue -u <user-name>
```

Note: Replace `user-name` with your cluster username
