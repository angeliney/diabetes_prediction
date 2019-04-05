To run all experiments, simply run conduct_exp_r.sh and conduct_exp_py.sh.

conduct_exp_r.sh: script to kick off all R jobs into cluster 
conduct_exp_py.sh: script to run all python jobs in a for loop

Each job run will output text file and rds/pickle file that contains the final trained model. 
The output text file summarizes the performance for that job. 
These text files can be concatenated with each other into 1 giant csv file, e.g. performance.csv.