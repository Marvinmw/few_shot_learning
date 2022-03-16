rm check.txt
find ./results -type f -name "log.txt" -print0 | while IFS= read -r -d '' filename
do
    echo $filename
    if grep -q 'CUDA' "$filename"
    then
        echo $filename >> check.txt
    fi
done


find ./results -type f -name "stat.json" -print0 | while IFS= read -r -d '' filename
do
    echo $filename
    d=$(dirname $filename)

    if [ ! -f $d/saved_model.pt ]
    then
        echo $filename >> check.txt
    fi
done