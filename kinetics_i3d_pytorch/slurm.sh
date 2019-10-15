#!/bin/bash
#SBATCH --job-name=CAN
#SBATCH --ntasks=1                  # Correr una sola tarea
#SBATCH --cpus=5
#SBATCH --output=log/job_%j.out    # Nombre del output (%j se reemplaza por el ID del trabajo)
#SBATCH --error=log/job_%j.err     # Output de errores (opcional)
#SBATCH --workdir=/mnt/nas2/GrimaRepo/jahurtado/codes/pytorch-mac-network/kinetics_i3d_pytorch   # Direccion donde correr el trabajo
#SBATCH --partition=ialab-high
#SBATCH --gres=gpu:1

pwd; hostname; date

source ../env/bin/activate

echo "Demo"
python demo_acc.py