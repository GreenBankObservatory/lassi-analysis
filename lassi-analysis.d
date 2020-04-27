#! /bin/bash
export LASSI_ANAL_HOME=/home/lassi/release
source $LASSI_ANAL_HOME/lassi-env/bin/activate
#exec python $LASSI_ANAL_HOME/lassi-analysis/server.py $*
cd $LASSI_ANAL_HOME/lassi-analysis
exec python server.py $*
