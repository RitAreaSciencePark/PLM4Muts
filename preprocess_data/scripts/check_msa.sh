train=$1
tests=$2
msas=($(awk 'BEGIN{FS=","}{if(NR==1)for(i=1;i<=NF;i++)if($i=="pdb_id")col=i;if(NR>1)print $col}' 'files/'$train'_for_'$tests | sort | uniq))
#msas=($(ls hhblits_files/utils_files/$train | grep "msa" ))

if [ -f files/$train'_no_msa' ]; then
	rm files/$train'_no_msa'
fi
if [ -f files/$train'_msa_not_found' ]; then
        rm files/$train'_msa_not_found'
fi
for p in "${msas[@]}"
do
	if [ -f 'hhblits_files/utils_files/'$train"/"$p".msa" ]; then
		size=$(wc -l hhblits_files/utils_files/$train"/"$p".msa" | awk '{print $1}')
		name=$(wc -l hhblits_files/utils_files/$train"/"$p".msa" | awk '{print $2}')

		if [ "$size" -lt 3 ];then
                	echo $p >> files/$train'_no_msa'
		fi
	else
		echo $p >> files/$train'_msa_not_found'
	fi
done
if [ -f files/$train'_no_msa' ]; then
	grep -v -f files/$train'_no_msa' files/$train'_for_'$tests > curated_data/$train'_for_'$tests'-clean.csv'
else
	cp files/$train'_for_'$tests curated_data/$train'_for_'$tests'-clean.csv'
	
fi
