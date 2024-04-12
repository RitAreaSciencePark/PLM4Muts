train=$1
tests=$2
label=$3

if [ $label = "train" ]; then
	db_path='curated_data/'
	db=$train"_for_"$tests"-clean.csv"
	data=$train
else
	db_path='curated_data/'
	db=$tests".csv"
	data=$tests
fi

row_path='hhblits_files/utils_files/'$data'/'
out_path='hhblits_files/'$data'/'
mkdir -p $out_path

sys=($(ls $row_path | grep ".msa"))
for s in "${sys[@]}"
do
	# check if msa is in path
	name=$(echo $s | sed 's/.msa//g')
	if [ -f $out_path$name ]; then
        	skip=1
	else
		# wild type msa only requires reformatting to get sequences in one line
		awk 'BEGIN{FS=""}{if($1==">"){if(NR==1)print $0; else {printf "\n";print $0;}}else printf toupper($0)}' $row_path$s > $out_path$name
	fi
done
