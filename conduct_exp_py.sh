#!/usr/bin/env bash
test_types=( merf deepsurv ) #merf is just rf for single visits
lags=( 0 3 )
booleans=( F T )
visit_types=( all first random )
train_thresh_years=( 1990 2000 2005 )

include_lab=T
cross_vals_fold=5
eq_train_ratio=T
cutoff=T
file_dir='/home/diabetes_prediction'

cd /home/diabetes_prediction
mkdir output
for include_ethdon in ${booleans[@]}; do
    for visit_type in ${visit_types[@]}; do
        for test_type in ${test_types[@]}; do
            for i in ${lags[@]}; do
                if [ "$visit_type" != "all" ] && [ "$i" -eq 1 ]; then
                    continue # only all visits need 1 lag
                fi
                if [ "$visit_type" == "first" ] && [ "$i" -gt 0 ]; then
                    continue # only 0 lag for first visits
                fi
                for train_thresh_year in ${train_thresh_years[@]}; do
                    output=ethdon$include_ethdon"_post"$train_thresh_year"_cutoff"$cutoff"_"$test_type"_"$visit_type"_lag"$i
                    if [ ! -f output/$output"_perf.pkl" ]; then
                        echo $output
                        inputs="$visit_type $i $include_lab $include_ethdon"
                        inputs="$inputs $eq_train_ratio output/$output $cross_vals_fold $train_thresh_year $cutoff $file_dir"
                        ~/miniconda2/bin/python python/conduct_$test_type.py $inputs
                    fi
                done
            done
        done
    done
done