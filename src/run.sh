#!/bin/bash
#SBATCH --job-name=hparam_tuning
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --output=./logs/slurm/training/%x_%j.out
#SBATCH --error=./logs/slurm/training/%x_%j.err
#SBATCH --mail-user=hiren.madhu@yale.edu                       # Email address to receive notifications
#SBATCH --mail-type=BEGIN                               # Email when the job starts
#SBATCH --mail-type=END                                 # Email when the job finishes
#SBATCH --mail-type=FAIL                                # Email if the job fails

cd /home/hm638/project_pi_sk2433/hm638/LLM-Assisted-Live-Coding/src
module load miniconda
conda init bash
conda activate music
# ---- paths / env ----
MAIN="impl_main.py"                  # change if your file lives elsewhere

# ---- hyperparameter grids (edit as you like) ----
hidden_dims=(256 512)
output_dims=(128 256)
num_layers=(3 5)
batch_sizes=(64 128)

lrs=(3e-3 1e-3)
wds=(1e-4 0)

num_epochs=(50)                     # keep shorter for sweeps
num_runs=(5)

schedulers=(cosine none)
warmup_ratios=(0.05 0.0)
min_lrs=(1e-5)

gpus=(0)                            # e.g., (0 1) to cycle across GPUs

# ---- sweep ----
exp_idx=0
total=$(( ${#hidden_dims[@]} * ${#output_dims[@]} * ${#num_layers[@]} * ${#batch_sizes[@]} * ${#lrs[@]} * ${#wds[@]} * ${#num_epochs[@]} * ${#num_runs[@]} * ${#schedulers[@]} * ${#warmup_ratios[@]} * ${#min_lrs[@]} ))

echo "Starting sweep with $total runs;"
for hd in "${hidden_dims[@]}"; do
  for od in "${output_dims[@]}"; do
    for nl in "${num_layers[@]}"; do
      for bs in "${batch_sizes[@]}"; do
        for lr in "${lrs[@]}"; do
          for wd in "${wds[@]}"; do
            for ne in "${num_epochs[@]}"; do
              for nr in "${num_runs[@]}"; do
                for sch in "${schedulers[@]}"; do
                  for wr in "${warmup_ratios[@]}"; do
                    for minlr in "${min_lrs[@]}"; do
                      gpu="${gpus[$((exp_idx % ${#gpus[@]}))]}"

                      # make a readable, filesystem-safe run name
                      run_name="hd_${hd}_od_${od}_nl_${nl}_bs_${bs}_lr_${lr}_wd_${wd}_ne_${ne}_nr_${nr}_sch_${sch}_wr_${wr}_minlr_${minlr}"
                      safe_name="$(echo "$run_name" | sed 's/\./p/g')"

                      echo "[$((exp_idx+1))/$total] -> $run_name (GPU $gpu)"
                      CUDA_VISIBLE_DEVICES="$gpu" python -u "$MAIN" \
                        --hidden_dim "$hd" \
                        --output_dim "$od" \
                        --num_layers "$nl" \
                        --batch_size "$bs" \
                        --lr "$lr" \
                        --wd "$wd" \
                        --num_epochs "$ne" \
                        --num_runs "$nr" \
                        --gpu "$gpu" \
                        --scheduler "$sch" \
                        --warmup_ratio "$wr" \
                        --min_lr "$minlr" \

                      exp_idx=$((exp_idx+1))
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
