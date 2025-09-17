To setup a conda environment in CSC Mahti/Puhti, please follow the instructions here[https://docs.csc.fi/computing/containers/tykky/].

Creating conda env in CSC Puhti/Mahti:
```bash 
module load tykky
mkdir /path/to/where/new/conda/env
conda-containerize new --prefix path/to/where/new/conda/env AVSegFormer/csc/env/env_avsegformer.yml
```

The post-install script needs to be run twice in CSC with certain lines uncommented at first. This is because the conda installation in Mahti/Puhti can only be modified inside the conda-containerize. So first run
```bash 
conda-containerize update /path/to/where/new/conda/env --post-install AVSegFormer/csc/env/env_avsegformer_post_install.sh
```

Then uncomment these lines in the post-instll file. Before running this, make sure you are connected to a node with GPUs: otherwise this will fail.
```bash
cd /scratch/project_2005102/sophie/repos/AVSegFormer/ops
sh make.sh
```
and run the post-install conda-containerize update command again.


Now your environment should work. Now download the necessary files following the above AVSegFromer README.

