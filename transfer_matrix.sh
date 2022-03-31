
for i in $(find milos_dataset/ -type f -name "mutationMatrix.csv" )
do
   dname=$(dirname $i)
   sed 's/,/ /g' $i > $dname/mutationMatrix_transfer.csv
   sed -i 's/MutantID/MUTERIA_MATRIX_KEY_COL/g' $dname/mutationMatrix_transfer.csv
done