import muteria.statistics.mutant_quality_indicators as mindicators
from muteria.statistics import algorithms
import  muteria.common.matrices as matrices
filename="mutationMatrix_transfer.csv"
matric = matrices.ExecutionMatrix(filename, )
result = matric.query_active_columns_of_rows()
print(mindicators.getSubsumingMutants(filename))