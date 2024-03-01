# User Guide for IFT 6289's Calcul Quebec Cluster

## Create an Account

Create your account for IFT6289's cluster here: [Account Signup](https://mokey.ift6759.calculquebec.cloud/auth/signup). The account will be automatically activated after registration.

## About the Cluster

This year **we share the same cluster with the IFT6759 (Advanced Machine Learning Projects) course**, so please don't be worried if you see IFT6759 in the host names!

Currently, out class has the resource allocations on Calcul Quebec clusters with 1 Resource Allocation Project Identifier: **def-sponsor00**. You will need to specify this project identifier in all your submitted jobs by the `--account=def-sponsor00` argument.

Through this project identifier, our class share the following resources:

* Storage:
  * `home/`: 300GB per user. While your home directory may seem like the logical place to store all your files and do all your work, in general this isn't the case; your home normally has a relatively small quota and **doesn't have especially good performance for writing and reading large amounts of data**. The most logical use of your home directory is typically source code, small parameter files and job submission scripts.
  * `project/`: 1TB shared by the two classes. The project space has a significantly **larger quota** and is well-adapted to **[sharing data](https://docs.alliancecan.ca/wiki/Sharing_data) among members of a research group** since it, unlike the home or scratch, is linked to a professor's account rather than an individual user. The data stored in the project space should be fairly static, that is to say the data are not likely to be changed many times in a month. Otherwise, frequently changing data, including just moving and renaming directories, in project can become a heavy burden on the tape-based backup system.
  * `scratch/`: 100GB per user. For **intensive read/write operations on large files (> 100 MB per file)**, scratch is the best choice. Remember however that important files must be copied off scratch since they are not backed up there, and older files are subject to [purging](https://docs.alliancecan.ca/wiki/Scratch_purging_policy). The scratch storage should therefore be used for temporary files: checkpoint files, output from jobs and other data that can easily be recreated.
* 1 Permanently Available Compute Node (Shared by the class):
  * 1/4 of a 32G Nvidia V100 GPU (8GB of GPU Memory)
  * 4 CPU Cores
  * 22GB of system memory

When submitted jobs require more resources, they can also be arranged. Our class, as a whole, can use up to 15 identical compute nodes mentioned above when needed. However, when the permanent node is used, allocation of new nodes may take more than 10 minutes to prepare. So plan your project ahead and wisely: do not wait until the last minute, or there will be insufficient computational resource!

## Login to Calcul Quebec

Use any SSH terminal to connect to the cluster:

``ssh username@ift6759.calculquebec.cloud``

You can setup keys for password-less entry. On your local machine, execute the following commands only once:

```shell
$ ssh-keygen
$ ssh-copy-id username@ift6759.calculquebec.cloud
```

## Transfer Files from Local Machine

### Option 1: Using the `scp ` command

You can use the `scp` command to transfer local files to CC clusters.

```shell
#download
scp username@ift6759.calculquebec.cloud:/home/username/projects/def-sponsor00/username/example.txt ~/PATH/TO/FOLDER/example_folder
#upload
scp ~/PATH/TO/FILE/example.txt username@ift6759.calculquebec.cloud:/home/username/projects/def-sponsor00/username/example_folder
```

### Option 2: Using VS Code's SSH Remote or Pycharm's Deployment

For VS Code's SSH Remote, please refer to [Developing on Remote Machines using SSH and Visual Studio Code](https://code.visualstudio.com/docs/remote/ssh).

For Pycharm's Deployment, please refer to [Tutorial: Deployment in PyCharm | PyCharm Documentation (jetbrains.com)](https://www.jetbrains.com/help/pycharm/tutorial-deployment-in-product.html). This tutorial is for an older version of Pycharm, but the process is similar to that of the current version of Pycharm. Please note that this feature requires the Pycharm Professional edition, which is [free for students]([Free Educational Licenses - Community Support (jetbrains.com)](https://www.jetbrains.com/community/education/#students)).

## Running Experiments

Calcul Quebec clusters use a workload management tool called Slurm to schedule jobs. Please refer to CC's wiki for detailed documentation: [Running jobs - CC Doc (alliancecan.ca)](https://docs.alliancecan.ca/wiki/Running_jobs). Slurm will distinguish servers in a cluster as either "login nodes" (provides the shell after you login or as data transfer nodes) and "compute nodes" (for running jobs). Do NOT run a job on the login node!

There are three common types of jobs:

* Normal jobs: submitted by `sbatch` with a job script. The cluster will run the job on a compute node in the background. Typically for running experiments.
* Interactive jobs: submitted by `salloc`. The cluster opens an interactive shell on a compute node. Typically for debugging on the cluster.
* JupyterLab jobs: submitted by accessing the [JupyterHub entrypoint](https://jupyter.ift6759.calculquebec.cloud/hub/login). Typically for interactive developments.

### Normal Jobs

You'll need a job script for a normal job. After having the script (e.g., `job_example.sh`), run the job with command `sbatch job_example.sh`. Below is a typical job script (`job_example.sh`):

```shell

#!/bin/bash
#SBATCH --gres=gpu:1                    # Request 1 GPU core. This takes up the complete computation node
#SBATCH --cpus-per-task=4  				# Request 4 CPU cores. This takes up the complete computation node
#SBATCH --mem=22000M       				# Memory proportional to GPUs: 22000M per GPU core
#SBATCH --time=0-03:00     				# DD-HH:MM:SS
#SBATCH --mail-user=example@gmail.com  	# Emails me when job starts, ends or fails
#SBATCH --mail-type=ALL
#SBATCH --account=def-sponsor00         # Resource Allocation Project Identifier

module load python/3.9 cuda cudnn scipy-stack

SOURCEDIR=~/ml-test

# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install packages on the virtualenv
pip install --no-index --upgrade pip tensorflow keras

# Prepare data
mkdir $SLURM_TMPDIR/data
tar xf ~/projects/def-sponsor00/data.tar -C $SLURM_TMPDIR/data

# Start training
python $SOURCEDIR/train.py $SLURM_TMPDIR/data
```

Run the job with `sbatch job_example.sh`. You will see a job ID after this command.

The workflow in a job script is as follows:

1. `#SBATCH` commands. These specify the GPU, CPU, system memory, running time, resource allocation projects, etc.
2. `module load`. Lots of commonly used libraries, e.g., Python (`python/3.x`), CUDA (`cuda`), CUDNN (`cudnn`), or even numpy+scipy+pandas+matplotlib+... (`scipy-stack`) are compiled as `modules` in CC. You can also specify the versions. Load them up to use them.
3. Prepare virtualenv. By using `$SLURM_TMPDIR`, you're creating a virtualenv on the compute node for better I/O performance.
4. Install python packages. If possible, keep using "`--no-index`" to prefer the prebuilt wheels on CC clusters.
5. Prepare data. By using `$SLURM_TMPDIR`, you're transferring the dataset to the compute node for better I/O performance.
6. Start training.

***Important: Dos and Don'ts***:

1. Do not use conda! Please use virtualenv instead. You don't even need to change the "Prepare virtualenv" part.
2. Do check available modules by `module avail` command. The compatibility of these provided modules are guaranteed. More about modules: [Using modules - CC Doc (alliancecan.ca)](https://docs.alliancecan.ca/wiki/Utiliser_des_modules/en)
3. Do use `--no-index` argument for `pip install`. This will search for pip wheels provided in the cluster. The compatibility of these provided wheels and the loaded modules are guaranteed. More about using python on the cluster: [Python - CC Doc (alliancecan.ca)](https://docs.alliancecan.ca/wiki/Python#Available_wheels)
4. Do use up the whole computation node. Jobs cannot share GPUs, but one node only contains one GPU, so jobs do not share nodes. Make sure that your hyperparameter setting (model size, batch size, etc.) can effectively use the resources in a compute node.


### Interactive Jobs

Start an interactive job by `salloc`:

```shell
$ salloc --time=1:0:0 --gres=gpu:1 --cpus-per-task=4 --mem=22000M --account=def-sponsor00
```

The parameters and rules are the same as normal jobs.

### Using JupyterLab on the cluster

Please submit your JupyterLab jobs through the [JupyterHub entrypoint](https://jupyter.ift6759.calculquebec.cloud/hub/login). You can configure the job time, number of cores, required memory size and GPU configurations through the entrypoint.

## Useful Commands

`squeue -u <username> [-t RUNNING] [-t PENDING]`: Check status of all submitted jobs.

`scancel <jobid>`: Kill a job.

`scancel -u <username>`: Kill all submitted jobs.

`srun --jobid <jobid> --pty tmux new-session -d 'htop -u $USER' \; split-window -h 'watch nvidia-smi' \; attach`: Monitoring CPU & GPU usage of a job.

`sshare -l -A def-sponsor00`: Check one's usage of `def-sponsor00` allocation. The more one uses the resource recently, the less priority (`LevelFS` shown in this command) one will be assigned.

## Useful Links:

Cluster Doc: [CC Doc (alliancecan.ca)](https://docs.alliancecan.ca/wiki/Technical_documentation)

Running jobs: [Running jobs - CC Doc (alliancecan.ca)](https://docs.alliancecan.ca/wiki/Running_jobs)

Python with the cluster: [Python - CC Doc (alliancecan.ca)](https://docs.alliancecan.ca/wiki/Python)

Machine learning with the cluster: [AI and Machine Learning - CC Doc (alliancecan.ca)](https://docs.alliancecan.ca/wiki/AI_and_Machine_Learning)
