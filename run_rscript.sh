args=( "$@" )
cd ${args[0]}
module load R
Rscript ${args[1]} ${args[@]:2}
