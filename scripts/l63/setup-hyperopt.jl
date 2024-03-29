import Pkg 
Pkg.activate(".")

using SlurmHyperopt, JLD2 

extra_calls = "echo \"------------------------------------------------------------\"
echo \"SLURM JOB ID: \$SLURM_JOBID\"
echo \"\$SLURM_NTASKS tasks\"
echo \"------------------------------------------------------------\"

module purge
module load julia/1.8.2
"

julia_call = "julia /p/tmp/maxgelbr/code/ChaoticNDETools.jl/scripts/l63/l63-hyperopt.jl \$SLURM_JOB_NAME \$SLURM_ARRAY_TASK_ID"

slurm_file = "l63-hyperpar.sh"

params = SlurmParams(qos="short", 
                    job_name="l63-hyperopt",
                    account="brasnet",
                    nodes=1, 
                    ntasks_per_node=8,
                    extra_calls=extra_calls,
                    parallel_jobs="10",
                    julia_call=julia_call,
                    file_path=slurm_file,
                    workdir="/p/tmp/maxgelbr/code/ChaoticNDETools.jl/scripts/l63")
                   
N_jobs = 70
sampler = RandomSampler(N_weights=5:30, N_hidden_layers=1:4, activation=["relu","swish"], τ_max=2:4, eta=[1f-2,1f-3,1f-4], eta_decrease=[5,10,15])
sho = SlurmHyperoptimizer(N_jobs, sampler, params)

JLD2.@save "hyperopt.jld2" sho