test_seq=$1
train_seq=$2
dir_code=$3
file_code=$4

out_dir='blastp/'$dir_code'/'
mkdir -p $out_dir

file=$out_dir$file_code

# blastp output has 3 columns: percent-identity, e-value, alignment-length

blastp -query <(echo $test_seq) -subject <(echo $train_seq) -outfmt "6 pident evalue length" -out $file 

# if no matches, blast output is empty
if [ -s $file ];then
	
	size=$(wc -l $file | awk '{print $1}')
	
	if [ "$size" -gt 1 ]; then
# if more than one alignment, select that with lowest evalue
		cat $file | sort -k2 -g  | head -n 1
	else
		cat $file
	fi
else
# write following if blast output is empty
	echo 0.0 10.0 0.0 > $file
fi
