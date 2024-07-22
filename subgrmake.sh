#!/bin/bash -l
#SBATCH --job-name=mkSBgraph
#SBATCH --nodes=1                      
#SBATCH --ntasks=4    
#SBATCH --gres=gpu:0
#SBATCH --time=1-0:00:00
#SBATCH --mem=100G
#SBATCH --partition=week

#SBATCH --output=sbmake.out
#SBATCH --error=sbmake.err

export PYTHONUNBUFFERED=TRUE
module load python-libs

python subgraphs.py --m sol_data_ESOL --d sol_data_ESOL
python subgraphs.py --m dock_hsprors --d dock_hsprors
python subgraphs.py --m dock_glprors --d dock_glprors
python subgraphs.py --m dock_ecprors --d dock_ecprors