#!/bin/bash

echo Good Started

echo starting CSV code

JID1=$(sbatch SaveToCSV.sh)   #This code calls a script that creates a CSV file in which I store all my results
JID1=${JID1##* }

echo Job ID is = $JID1

JID2=$(sbatch --dependency=afterany:$JID1 NewWindow.sh) #This script runs the main code and it only runs once the above called script finishes
JID3=${JID2##* }
echo Job ID is = $JID3
sbatch --dependency=afterany:$JID3 Print.sh #This one is just waiting for all the codes to finish up so that it could send me an email that codes have finished running and I could time how long it took
echo Dependency ran
