for f in $(find $1 -type f -name "few_shot_test_pair.json")
do
	echo $f
	d=$(dirname $f)
	h=$(dirname $d)
	cp $f $h
	rm $f
done
