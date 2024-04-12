#!/bin/bash

dataset=$1
sys=($(ls 'hhblits_files/queries/'$dataset))

found_uniclust=$(ls ./ | grep "uniclust30_2018_08")
if [ -z "$found_uniclust" ];then
        wget http://gwdu111.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08.tar.gz
        tar -xvzf uniclust30_2018_08.tar.gz
fi

database='uniclust30_2018_08/uniclust30_2018_08'
stuff='hhblits_files/utils_files/'$dataset'/'
mkdir -p $stuff	

for s in "${sys[@]}"
do

	if [ -f $stuff$s'.msa' ]; then
		skip=1
	else

		query='hhblits_files/queries/'$dataset'/'$s

		hhblits -v 1 -cpu 2 -i $query -o $stuff$s'.hhr' -oa3m $stuff$s'.a3m' -n 3 -d $database 

		perl 'scripts/reformat.pl' -v 1 -r a3m fas $stuff$s.a3m $stuff$s'.msa'
	fi
done
