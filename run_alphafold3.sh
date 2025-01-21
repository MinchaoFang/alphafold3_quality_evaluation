#!/bin/bash
#SBATCH --job-name=alphafold3_singularity
#SBATCH -c 1
#SBATCH -p caolx-a40,a100-40g,a40-quad,a40-tmp
#SBATCH --gres=gpu:1
#SBATCH -q gpu-huge
#SBATCH --mem=40G

module load alphafold/3_a40-tmp
singularity exec \
        --nv \
        --bind /storage/caolongxingLab/fangminchao/work/alphafold3_quality_evaluation/output_test:/root/output \
        --bind /storage/caolongxingLab/fangminchao/ \
        --bind /storage/caolongxingLab/fangminchao/tools/alphafold3/model:/root/models \
        --bind /storage/caolongxingLab/fangminchao/database/AF3/public_databases:/root/public_databases \
        /soft/bio/alphafold/3/alphafold3.sif \
python /storage/caolongxingLab/fangminchao/work/alphafold3_quality_evaluation/run_alphafold_avail.py \
        --json_path=/storage/caolongxingLab/fangminchao/work/alphafold3_quality_evaluation/input_CD3_2.json  \
        --ref_pdb_path=/storage/caolongxingLab/fangminchao/work/alphafold3_quality_evaluation/positions_out_outside.pkl \
        --ref_time_steps=50 \
        --model_dir=/root/models \
        --db_dir=/root/public_databases \
        --output_dir=/root/output \
        --jax_compilation_cache_dir=/root/output/jax_compilation \
        --norun_data_pipeline


#--ref_pdb_path=/storage/caolongxingLab/fangminchao/work/alphafold3_quality_evaluation/positions_out_outside.pkl \
   #     --ref_time_steps=10 \
#        --ref_pdb_path=/storage/caolongxingLab/fangminchao/work/alphafold3_quality_evaluation/positions_out_outside.pkl \
  #      --ref_time_steps=30 \
##         --ref_pkl_dump_path=/storage/caolongxingLab/fangminchao/work/alphafold3_quality_evaluation/positions_out_outside_test.pkl \