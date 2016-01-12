#!/bin/bash
m=10
n=100
populations=25

# Generate datasets
for vr in 0 1 2; do
    for phi in 0.3 0.8 1.0; do
	for ap in 0.9; do
	    (python2.7 ./dpopt.py -m $m -n $n -disp $phi -ap $ap -pop $populations -vr $vr -gd ;
	     python2.7 ./dpopt.py -m $m -n $n -disp $phi -ap $ap -pop $populations -vr $vr -whatif) &
	done
    done
    wait
done
# Run experiments
for phi in 0.3 0.8 1.0; do
    for met in 0 1 2; do
	for ap in 0.9 0.5 0.3; do
	    for vr in 0 1 2; do
		logf=log-m$m-n$n-pop$populations-disp$phi-ap$ap-vr$vr-method$met
		echo time python2.7 ./dpopt.py -m $m -n $n -disp $phi -ap $ap -pop $populations -vr $vr -method $met to $logf.log
		((time python2.7 ./dpopt.py -m $m -n $n -disp $phi -ap $ap -pop $populations -vr $vr -method $met; echo DONE > $logf.done) > $logf.log 2>&1) &
	    done
	done
	wait
    done
done

