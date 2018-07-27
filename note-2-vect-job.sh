#!/bin/bash -l

#SBATCH -N 1         #Use 1 nodes
#SBATCH -t 01:00:00  #Set 1 hour time limit
#SBATCH -q regular   #Use the regular QOS
#SBATCH -L project   #Job requires $project file system
#SBATCH -C haswell   #Use KNL nodes in quad cache format (default, recommended)
#SBATCH -D /project/projectdirs/m1532/rafael/nlp_suicide   #Working directory

module load python/3.6-anaconda-4.4
source activate myenv
python Notes2Vect.py 100&
python Notes2Vect.py 48&
python Notes2Vect.py 20&

wait

echo "Done with all 3 embeddings"
