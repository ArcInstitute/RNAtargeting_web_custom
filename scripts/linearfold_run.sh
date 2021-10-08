#!/bin/bash
#
#SBATCH -p gpu
#SBATCH --job-name=refseq_linearfold_guide
#SBATCH --output=efseq_linearfold_guide.txt
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2
#SBATCH --time=1-23:59:59
##SBATCH --gpus 2
#SBATCH --cpus-per-gpu=10
##SBATCH --mem=256000
#SBATCH --mem-per-gpu=24G

echo "start!"

cd LinearFold

cat ../dataset/mcherry_linfold_guides_input.fasta | ./linearfold  > ../dataset/mcherry_linfold_guides_output.txt
cat ../dataset/mcherry_linfold_guides_nearby15_input.fasta | ./linearfold  > ../dataset/mcherry_linfold_target_flank15_output.txt
cat ../dataset/mcherry_linfold_guides_constraints_nearby15_input.fasta | ./linearfold --constraints > ../dataset/mcherry_linfold_constraints_target_flank15_output.txt

echo "done!"
