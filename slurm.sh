#!/bin/bash
#SBATCH --job-name=CAN
#SBATCH --ntasks=1                  # Correr una sola tarea
#SBATCH --cpus=5
#SBATCH --output=job_%j.out    # Nombre del output (%j se reemplaza por el ID del trabajo)
#SBATCH --error=job_%j.err     # Output de errores (opcional)
#SBATCH --workdir=/mnt/nas2/GrimaRepo/jahurtado/codes/pytorch-mac-network   # Direccion donde correr el trabajo
#SBATCH --nodelist=scylla
#SBATCH --gres=gpu:Geforce-RTX:1
#SBATCH --partition=ialab-high

pwd; hostname; date

echo "Inicio de evaluacion Actions Transformations"

source ./env/bin/activate
export COMET_API_KEY=3CY3z4b2eYk08ZWoVOW912Yfl
export COMET_DISABLE=False

echo "More Step"
python3 code/main.py --cfg cfg/s2s_train_mac_other_node_step.yml --manualSeed 342

echo "More Context Learned"
python3 code/main.py --cfg cfg/s2s_train_mac_other_node_learned.yml --manualSeed 342

echo "Model with more dimension"
python3 code/main.py --cfg cfg/s2s_train_mac_other_node_dim_module.yml --manualSeed 342
echo "Finished with job $SLURM_JOBID"















#!/bin/bash
#SBATCH --job-name=CAN
#SBATCH --ntasks=1                  # Correr una sola tarea
#SBATCH --cpus=5
#SBATCH --output=job_%j.out    # Nombre del output (%j se reemplaza por el ID del trabajo)
#SBATCH --error=job_%j.err     # Output de errores (opcional)
#SBATCH --workdir=/mnt/nas/GrimaRepo/aespinosa/pytorch-mac-network   # Direccion donde correr el trabajo
#SBATCH --nodelist=hydra
# SBATCH --gres=gpu:Geforce-GTX:1
#SBATCH --gres=gpu:1
#SBATCH --partition=ialab-high

pwd; hostname; date

echo "Inicio de evaluacion Actions Transformations"

#source actions_transformations_venv/bin/activate
#export CUDA_VISIBLE_DEVICES=6
source /mnt/nas2/GrimaRepo/aespinosa/envs/datascience/bin/activate
export COMET_API_KEY=3CY3z4b2eYk08ZWoVOW912Yfl
python3 code/main.py --cfg cfg/s2s_train_mac_other_node.yml --manualSeed 342
echo "Finished with job $SLURM_JOBID"
