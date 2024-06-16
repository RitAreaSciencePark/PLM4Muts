
dir_code=$1
blast_path='blastp/'
out_path='blastp/'
mkdir -p $out_path
all_files=($(ls $blast_path$dir_code))

if [ -f $blast_path$dir_code'_merged_blast' ]; then
	mv $blast_path$dir_code'_merged_blast' $blast_path'old_'$dir_code'_merged_blast'
fi

echo 'code' 'identity' 'evalue' 'length' > $blast_path$dir_code'_merged_blast'
for code in "${all_files[@]}"
do
	line=$(head -n 1 $blast_path$dir_code'/'$code)
	identity=$(echo $line | awk '{print $1}')
	evalue=$(echo $line | awk '{print $2}')
	overlap=$(echo $line | awk '{print $3}')
	
	echo $code $identity $evalue $overlap >> $blast_path$dir_code'_merged_blast'
done
#rm -r $blast_path$dir_code
