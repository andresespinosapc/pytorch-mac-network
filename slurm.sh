#!/bin/bash
#SBATCH --job-name=CAN
#SBATCH -t 3-2:00                    # tiempo maximo en el cluster (D-HH:MM)
#SBATCH --ntasks=1                  # Correr una sola tarea
#SBATCH --cpus=5
#SBATCH --output=log/job_%j.out    # Nombre del output (%j se reemplaza por el ID del trabajo)
#SBATCH --error=log/job_%j.err     # Output de errores (opcional)
#SBATCH --workdir=/mnt/nas2/GrimaRepo/jahurtado/codes/pytorch-mac-network   # Direccion donde correr el trabajo
#SBATCH --partition=ialab-high
#SBATCH --nodelist=scylla
#SBATCH --gres=gpu:Geforce-GTX:1

pwd; hostname; date

echo "Inicio de evaluacion Actions Transformations"

source ./env/bin/activate
export COMET_API_KEY=3CY3z4b2eYk08ZWoVOW912Yfl
export COMET_DISABLE=1

echo "More Context Learned"
python3 code/main.py --cfg cfg/s2s_train_mac.yml --manualSeed 342

echo "Finished with job $SLURM_JOBID"
