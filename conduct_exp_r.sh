#!/usr/bin/env bash
test_types_all_visits=( mlm coxph coxphtime aft glmnetcox rfsrc )
test_types_single_visit=( glm glmnet coxph aft glmnetcox rfsrc )
lags=( 0 3 )
booleans=( F T )
visit_types=( all first random )
train_thresh_years=( 1990 2000 2005 )
rscript="./run_rscript.sh"

include_lab=T
eq_train_ratio=T
cross_vals_fold=5
cutoff=T

cd /home/diabetes_prediction
mkdir output
mkdir output/oe
for include_ethdon in ${booleans[@]}; do
    for train_thresh_year in ${train_thresh_years[@]}; do
        for visit_type in ${visit_types[@]}; do
            if [ "$visit_type" == "all" ]; then
                test_types=${test_types_all_visits[@]}
            else
                test_types=${test_types_single_visit[@]}
            fi
            for test_type in ${test_types[@]}; do
                for i in ${lags[@]}; do
                    if [ "$visit_type" != "all" ] && [ "$i" -eq 1 ]; then
                        continue # only all visits need 1 lag
                    fi
                    if [ "$visit_type" == "first" ] && [ "$i" -gt 0 ]; then
                        continue # only all visits need 1 lag
                    fi
                    output=ethdon$include_ethdon"_post"$train_thresh_year"_cutoff"$cutoff"_"$test_type"_"$visit_type"_lag"$i
                    if [ ! -f  output/$output".txt" ]; then
                        echo $output
                        rm output/oe/$output.e output/oe/$output.o output/$output.txt output/$output"_perf.rds"
                        inputs="$PWD ./R/conduct_exp.R $test_type $visit_type $i $include_lab $include_ethdon"
                        inputs="$inputs $eq_train_ratio output/$output $cross_vals_fold $train_thresh_year $cutoff"
                        qsub -d $PWD -o output/oe/$output.o -e output/oe/$output.e -l vmem=48g,walltime=24:00:00 -F "$inputs" $rscript
                    fi
                done
            done
        done
    done
done
