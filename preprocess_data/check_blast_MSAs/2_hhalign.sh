
train=$1
tests=$2
vals=($(cat parsed/$train"-"$tests"_pairs" | awk '{if(NR>1)print $0}'))

qpath="../"$train"/test/MSA_"$tests"/"
dpath="../"$train"/train/MSA_"$train"/"
mkdir -p $train"-"$tests

if [ -s parsed/parsed_$train-$tests ];then
	rm parsed/parsed_$train-$tests
fi
echo "code" "evalue" "identity" "overlap" "neff" >> parsed/parsed_$train-$tests

for v in "${vals[@]}"
do
	query=$(echo $v | awk 'BEGIN{FS=","}{print $1}')
	db=$(echo $v | awk 'BEGIN{FS=","}{print $2}')
	hhalign -i $qpath$query -t $dpath$db -o $train"-"$tests"/"$query"-"$db
	row=$(grep Identities $train"-"$tests"/"$query"-"$db)
        evalue=$(echo $row | awk '{print $2}' | sed 's/E-value=//g')	
	id=$(echo $row |  awk '{print $5}' | sed 's/Identities=//g' | sed 's/\%//g')
	len=$(echo $row | awk '{print $4}' | sed 's/Aligned_cols=//g')
	test_len=$(awk 'BEGIN{FS=""}{if(NR==2)print NF}' $qpath$query)
	over=$(echo $len | awk -v tl=$test_len '{print $1/tl}')
	neff=$(echo $row | awk '{print $8}' | sed 's/Template_Neff=//g')
	echo $v $evalue $id $over $neff >> parsed/parsed_$train-$tests
done
