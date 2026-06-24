#!/bin/bash
#SBATCH --job-name bmlike
#SBATCH --cpus-per-task=14
#SBATCH --mem 16G
#SBATCH --partition cpu-large
##SBATCH --output HPFit/experiments/2026_04_mio/results/log/slurm-%A_%a.out
#SBATCH --time=09:00:00
##SBATCH --array=1-130
#SBATCH --array=1-2
## allow 1 hour for each model below, then some.

unset MATLABPATH
module load matlab/R2024a 2>/dev/null
module load R/4.4.1
module load gurobi/13.0.0

#GRB_LICENSE_FILE=$HOME/gurobi.lic

FOLNAME=bm-like # folder where data is
EXP=2026_04_mio # experiment
JOBNAME=mio # name of job on SGE
TIMELIMIT=60
#TIMELIMIT=3600
Q=0.50
DEP_VAR=TRUE

SRCLOC=$HOME/HPFit/experiments/src
DATALOC=$HOME/HPFit/experiments/$EXP/data/$FOLNAME
RESLOC=$HOME/HPFit/experiments/$EXP/results/$JOBNAME/$FOLNAME
SEEDFILE=$DATALOC/$FOLNAME.in
MOSEKLOC=$HOME/src/mosek/9.3/toolbox/r2015a
GUROBILOC=/opt/gurobi1300/linux64/matlab
mkdir -p $RESLOC # make folder for results
mkdir -p $RESLOC/log # make folder for logs

ID=$SLURM_ARRAY_TASK_ID

SEED=$(sed -n -e "$ID p" $SEEDFILE)
echo "library(MASS)" > $RESLOC/hyper.$ID.in
echo "library(matlabr)" >> $RESLOC/hyper.$ID.in
echo "options(matlab.path='/opt/matlab2024a/bin')" >> $RESLOC/hyper.$ID.in
echo "source(\"$SRCLOC/run_mio.R\")" >> $RESLOC/hyper.$ID.in
echo "set.seed(12345)" >> $RESLOC/hyper.$ID.in
echo "run_mio(\"$DATALOC\", \"$SRCLOC\", \"$SEED\", $Q, $DEP_VAR, \"mio1\",        $TIMELIMIT, \"$RESLOC\", FALSE, \"$MOSEKLOC\", \"$GUROBILOC\")" >> $RESLOC/hyper.$ID.in
#echo "set.seed(12345)" >> $RESLOC/hyper.$ID.in
#echo "run_mio(\"$DATALOC\", \"$SRCLOC\", \"$SEED\", $Q, $DEP_VAR, \"mio-bm\",      $TIMELIMIT, \"$RESLOC\", FALSE, \"$MOSEKLOC\", \"$GUROBILOC\")" >> $RESLOC/hyper.$ID.in
#echo "set.seed(12345)" >> $RESLOC/hyper.$ID.in
#echo "run_mio(\"$DATALOC\", \"$SRCLOC\", \"$SEED\", $Q, $DEP_VAR, \"alg3-mio-bm\", $TIMELIMIT, \"$RESLOC\", FALSE, \"$MOSEKLOC\", \"$GUROBILOC\")" >> $RESLOC/hyper.$ID.in
#echo "set.seed(12345)" >> $RESLOC/hyper.$ID.in
#echo "run_mio(\"$DATALOC\", \"$SRCLOC\", \"$SEED\", $Q, $DEP_VAR, \"alg3-mio1\",   $TIMELIMIT, \"$RESLOC\", FALSE, \"$MOSEKLOC\", \"$GUROBILOC\")" >> $RESLOC/hyper.$ID.in
#echo "set.seed(12345)" >> $RESLOC/hyper.$ID.in
#echo "run_mio(\"$DATALOC\", \"$SRCLOC\", \"$SEED\", $Q, $DEP_VAR, \"lqs-mio1\",    $TIMELIMIT, \"$RESLOC\", TRUE,  \"$MOSEKLOC\", \"$GUROBILOC\")" >> $RESLOC/hyper.$ID.in
#echo "set.seed(12345)" >> $RESLOC/hyper.$ID.in
#echo "run_mio(\"$DATALOC\", \"$SRCLOC\", \"$SEED\", $Q, $DEP_VAR, \"lqs-mio-bm\",  $TIMELIMIT, \"$RESLOC\", TRUE,  \"$MOSEKLOC\", \"$GUROBILOC\")" >> $RESLOC/hyper.$ID.in
#echo "set.seed(12345)" >> $RESLOC/hyper.$ID.in
#echo "run_mio(\"$DATALOC\", \"$SRCLOC\", \"$SEED\", $Q, $DEP_VAR, \"cbq-mio1\",    $TIMELIMIT, \"$RESLOC\", FALSE, \"$MOSEKLOC\", \"$GUROBILOC\")" >> $RESLOC/hyper.$ID.in
#echo "set.seed(12345)" >> $RESLOC/hyper.$ID.in
#echo "run_mio(\"$DATALOC\", \"$SRCLOC\", \"$SEED\", $Q, $DEP_VAR, \"cbq-mio-bm\",  $TIMELIMIT, \"$RESLOC\", FALSE, \"$MOSEKLOC\", \"$GUROBILOC\")" >> $RESLOC/hyper.$ID.in

/opt/R-4.4.1/bin/R CMD BATCH $RESLOC/hyper.$ID.in $RESLOC/log/$ID.Rout

rm $RESLOC/hyper.$ID.in
#rm $RESLOC/log/$ID.Rout
rm $HOME/HPFit/experiments/$EXP/scripts/slurm-${SLURM_ARRAY_JOB_ID}_$ID.out
