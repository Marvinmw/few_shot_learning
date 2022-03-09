#!/bin/bash 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_24/fom_export/ -o relevance_java_dot/commons-lang/lang_24/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_24/original/ -o relevance_java_dot/commons-lang/lang_24/original 
cp relevance_java_bytecode/commons-lang/lang_24/mutants_info.json relevance_java_dot/commons-lang/lang_24/ 
cp relevance_java_bytecode/commons-lang/lang_24/interaction_pairs.csv relevance_java_dot/commons-lang/lang_24/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_5/fom_export/ -o relevance_java_dot/commons-lang/lang_5/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_5/original/ -o relevance_java_dot/commons-lang/lang_5/original 
cp relevance_java_bytecode/commons-lang/lang_5/mutants_info.json relevance_java_dot/commons-lang/lang_5/ 
cp relevance_java_bytecode/commons-lang/lang_5/interaction_pairs.csv relevance_java_dot/commons-lang/lang_5/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_18/fom_export/ -o relevance_java_dot/commons-lang/lang_18/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_18/original/ -o relevance_java_dot/commons-lang/lang_18/original 
cp relevance_java_bytecode/commons-lang/lang_18/mutants_info.json relevance_java_dot/commons-lang/lang_18/ 
cp relevance_java_bytecode/commons-lang/lang_18/interaction_pairs.csv relevance_java_dot/commons-lang/lang_18/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_6/fom_export/ -o relevance_java_dot/commons-lang/lang_6/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_6/original/ -o relevance_java_dot/commons-lang/lang_6/original 
cp relevance_java_bytecode/commons-lang/lang_6/mutants_info.json relevance_java_dot/commons-lang/lang_6/ 
cp relevance_java_bytecode/commons-lang/lang_6/interaction_pairs.csv relevance_java_dot/commons-lang/lang_6/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_16/fom_export/ -o relevance_java_dot/commons-lang/lang_16/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_16/original/ -o relevance_java_dot/commons-lang/lang_16/original 
cp relevance_java_bytecode/commons-lang/lang_16/mutants_info.json relevance_java_dot/commons-lang/lang_16/ 
cp relevance_java_bytecode/commons-lang/lang_16/interaction_pairs.csv relevance_java_dot/commons-lang/lang_16/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_2/fom_export/ -o relevance_java_dot/commons-lang/lang_2/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_2/original/ -o relevance_java_dot/commons-lang/lang_2/original 
cp relevance_java_bytecode/commons-lang/lang_2/mutants_info.json relevance_java_dot/commons-lang/lang_2/ 
cp relevance_java_bytecode/commons-lang/lang_2/interaction_pairs.csv relevance_java_dot/commons-lang/lang_2/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_25/fom_export/ -o relevance_java_dot/commons-lang/lang_25/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_25/original/ -o relevance_java_dot/commons-lang/lang_25/original 
cp relevance_java_bytecode/commons-lang/lang_25/mutants_info.json relevance_java_dot/commons-lang/lang_25/ 
cp relevance_java_bytecode/commons-lang/lang_25/interaction_pairs.csv relevance_java_dot/commons-lang/lang_25/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_27/fom_export/ -o relevance_java_dot/commons-lang/lang_27/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_27/original/ -o relevance_java_dot/commons-lang/lang_27/original 
cp relevance_java_bytecode/commons-lang/lang_27/mutants_info.json relevance_java_dot/commons-lang/lang_27/ 
cp relevance_java_bytecode/commons-lang/lang_27/interaction_pairs.csv relevance_java_dot/commons-lang/lang_27/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_23/fom_export/ -o relevance_java_dot/commons-lang/lang_23/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_23/original/ -o relevance_java_dot/commons-lang/lang_23/original 
cp relevance_java_bytecode/commons-lang/lang_23/mutants_info.json relevance_java_dot/commons-lang/lang_23/ 
cp relevance_java_bytecode/commons-lang/lang_23/interaction_pairs.csv relevance_java_dot/commons-lang/lang_23/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_30/fom_export/ -o relevance_java_dot/commons-lang/lang_30/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_30/original/ -o relevance_java_dot/commons-lang/lang_30/original 
cp relevance_java_bytecode/commons-lang/lang_30/mutants_info.json relevance_java_dot/commons-lang/lang_30/ 
cp relevance_java_bytecode/commons-lang/lang_30/interaction_pairs.csv relevance_java_dot/commons-lang/lang_30/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_13/fom_export/ -o relevance_java_dot/commons-lang/lang_13/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_13/original/ -o relevance_java_dot/commons-lang/lang_13/original 
cp relevance_java_bytecode/commons-lang/lang_13/mutants_info.json relevance_java_dot/commons-lang/lang_13/ 
cp relevance_java_bytecode/commons-lang/lang_13/interaction_pairs.csv relevance_java_dot/commons-lang/lang_13/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_9/fom_export/ -o relevance_java_dot/commons-lang/lang_9/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_9/original/ -o relevance_java_dot/commons-lang/lang_9/original 
cp relevance_java_bytecode/commons-lang/lang_9/mutants_info.json relevance_java_dot/commons-lang/lang_9/ 
cp relevance_java_bytecode/commons-lang/lang_9/interaction_pairs.csv relevance_java_dot/commons-lang/lang_9/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_26/fom_export/ -o relevance_java_dot/commons-lang/lang_26/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_26/original/ -o relevance_java_dot/commons-lang/lang_26/original 
cp relevance_java_bytecode/commons-lang/lang_26/mutants_info.json relevance_java_dot/commons-lang/lang_26/ 
cp relevance_java_bytecode/commons-lang/lang_26/interaction_pairs.csv relevance_java_dot/commons-lang/lang_26/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_29/fom_export/ -o relevance_java_dot/commons-lang/lang_29/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_29/original/ -o relevance_java_dot/commons-lang/lang_29/original 
cp relevance_java_bytecode/commons-lang/lang_29/mutants_info.json relevance_java_dot/commons-lang/lang_29/ 
cp relevance_java_bytecode/commons-lang/lang_29/interaction_pairs.csv relevance_java_dot/commons-lang/lang_29/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_1/fom_export/ -o relevance_java_dot/commons-lang/lang_1/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_1/original/ -o relevance_java_dot/commons-lang/lang_1/original 
cp relevance_java_bytecode/commons-lang/lang_1/mutants_info.json relevance_java_dot/commons-lang/lang_1/ 
cp relevance_java_bytecode/commons-lang/lang_1/interaction_pairs.csv relevance_java_dot/commons-lang/lang_1/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_4/fom_export/ -o relevance_java_dot/commons-lang/lang_4/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_4/original/ -o relevance_java_dot/commons-lang/lang_4/original 
cp relevance_java_bytecode/commons-lang/lang_4/mutants_info.json relevance_java_dot/commons-lang/lang_4/ 
cp relevance_java_bytecode/commons-lang/lang_4/interaction_pairs.csv relevance_java_dot/commons-lang/lang_4/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_31/fom_export/ -o relevance_java_dot/commons-lang/lang_31/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_31/original/ -o relevance_java_dot/commons-lang/lang_31/original 
cp relevance_java_bytecode/commons-lang/lang_31/mutants_info.json relevance_java_dot/commons-lang/lang_31/ 
cp relevance_java_bytecode/commons-lang/lang_31/interaction_pairs.csv relevance_java_dot/commons-lang/lang_31/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_14/fom_export/ -o relevance_java_dot/commons-lang/lang_14/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_14/original/ -o relevance_java_dot/commons-lang/lang_14/original 
cp relevance_java_bytecode/commons-lang/lang_14/mutants_info.json relevance_java_dot/commons-lang/lang_14/ 
cp relevance_java_bytecode/commons-lang/lang_14/interaction_pairs.csv relevance_java_dot/commons-lang/lang_14/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_28/fom_export/ -o relevance_java_dot/commons-lang/lang_28/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_28/original/ -o relevance_java_dot/commons-lang/lang_28/original 
cp relevance_java_bytecode/commons-lang/lang_28/mutants_info.json relevance_java_dot/commons-lang/lang_28/ 
cp relevance_java_bytecode/commons-lang/lang_28/interaction_pairs.csv relevance_java_dot/commons-lang/lang_28/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_12/fom_export/ -o relevance_java_dot/commons-lang/lang_12/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_12/original/ -o relevance_java_dot/commons-lang/lang_12/original 
cp relevance_java_bytecode/commons-lang/lang_12/mutants_info.json relevance_java_dot/commons-lang/lang_12/ 
cp relevance_java_bytecode/commons-lang/lang_12/interaction_pairs.csv relevance_java_dot/commons-lang/lang_12/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_21/fom_export/ -o relevance_java_dot/commons-lang/lang_21/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_21/original/ -o relevance_java_dot/commons-lang/lang_21/original 
cp relevance_java_bytecode/commons-lang/lang_21/mutants_info.json relevance_java_dot/commons-lang/lang_21/ 
cp relevance_java_bytecode/commons-lang/lang_21/interaction_pairs.csv relevance_java_dot/commons-lang/lang_21/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_10/fom_export/ -o relevance_java_dot/commons-lang/lang_10/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_10/original/ -o relevance_java_dot/commons-lang/lang_10/original 
cp relevance_java_bytecode/commons-lang/lang_10/mutants_info.json relevance_java_dot/commons-lang/lang_10/ 
cp relevance_java_bytecode/commons-lang/lang_10/interaction_pairs.csv relevance_java_dot/commons-lang/lang_10/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_17/fom_export/ -o relevance_java_dot/commons-lang/lang_17/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_17/original/ -o relevance_java_dot/commons-lang/lang_17/original 
cp relevance_java_bytecode/commons-lang/lang_17/mutants_info.json relevance_java_dot/commons-lang/lang_17/ 
cp relevance_java_bytecode/commons-lang/lang_17/interaction_pairs.csv relevance_java_dot/commons-lang/lang_17/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_11/fom_export/ -o relevance_java_dot/commons-lang/lang_11/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_11/original/ -o relevance_java_dot/commons-lang/lang_11/original 
cp relevance_java_bytecode/commons-lang/lang_11/mutants_info.json relevance_java_dot/commons-lang/lang_11/ 
cp relevance_java_bytecode/commons-lang/lang_11/interaction_pairs.csv relevance_java_dot/commons-lang/lang_11/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_19/fom_export/ -o relevance_java_dot/commons-lang/lang_19/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_19/original/ -o relevance_java_dot/commons-lang/lang_19/original 
cp relevance_java_bytecode/commons-lang/lang_19/mutants_info.json relevance_java_dot/commons-lang/lang_19/ 
cp relevance_java_bytecode/commons-lang/lang_19/interaction_pairs.csv relevance_java_dot/commons-lang/lang_19/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_7/fom_export/ -o relevance_java_dot/commons-lang/lang_7/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_7/original/ -o relevance_java_dot/commons-lang/lang_7/original 
cp relevance_java_bytecode/commons-lang/lang_7/mutants_info.json relevance_java_dot/commons-lang/lang_7/ 
cp relevance_java_bytecode/commons-lang/lang_7/interaction_pairs.csv relevance_java_dot/commons-lang/lang_7/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_15/fom_export/ -o relevance_java_dot/commons-lang/lang_15/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_15/original/ -o relevance_java_dot/commons-lang/lang_15/original 
cp relevance_java_bytecode/commons-lang/lang_15/mutants_info.json relevance_java_dot/commons-lang/lang_15/ 
cp relevance_java_bytecode/commons-lang/lang_15/interaction_pairs.csv relevance_java_dot/commons-lang/lang_15/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_3/fom_export/ -o relevance_java_dot/commons-lang/lang_3/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_3/original/ -o relevance_java_dot/commons-lang/lang_3/original 
cp relevance_java_bytecode/commons-lang/lang_3/mutants_info.json relevance_java_dot/commons-lang/lang_3/ 
cp relevance_java_bytecode/commons-lang/lang_3/interaction_pairs.csv relevance_java_dot/commons-lang/lang_3/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_20/fom_export/ -o relevance_java_dot/commons-lang/lang_20/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-lang/lang_20/original/ -o relevance_java_dot/commons-lang/lang_20/original 
cp relevance_java_bytecode/commons-lang/lang_20/mutants_info.json relevance_java_dot/commons-lang/lang_20/ 
cp relevance_java_bytecode/commons-lang/lang_20/interaction_pairs.csv relevance_java_dot/commons-lang/lang_20/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_9/fom_export/ -o relevance_java_dot/commons-io/io_9/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_9/original/ -o relevance_java_dot/commons-io/io_9/original 
cp relevance_java_bytecode/commons-io/io_9/mutants_info.json relevance_java_dot/commons-io/io_9/ 
cp relevance_java_bytecode/commons-io/io_9/interaction_pairs.csv relevance_java_dot/commons-io/io_9/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_1/fom_export/ -o relevance_java_dot/commons-io/io_1/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_1/original/ -o relevance_java_dot/commons-io/io_1/original 
cp relevance_java_bytecode/commons-io/io_1/mutants_info.json relevance_java_dot/commons-io/io_1/ 
cp relevance_java_bytecode/commons-io/io_1/interaction_pairs.csv relevance_java_dot/commons-io/io_1/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_6/fom_export/ -o relevance_java_dot/commons-io/io_6/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_6/original/ -o relevance_java_dot/commons-io/io_6/original 
cp relevance_java_bytecode/commons-io/io_6/mutants_info.json relevance_java_dot/commons-io/io_6/ 
cp relevance_java_bytecode/commons-io/io_6/interaction_pairs.csv relevance_java_dot/commons-io/io_6/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_23/fom_export/ -o relevance_java_dot/commons-io/io_23/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_23/original/ -o relevance_java_dot/commons-io/io_23/original 
cp relevance_java_bytecode/commons-io/io_23/mutants_info.json relevance_java_dot/commons-io/io_23/ 
cp relevance_java_bytecode/commons-io/io_23/interaction_pairs.csv relevance_java_dot/commons-io/io_23/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_12/fom_export/ -o relevance_java_dot/commons-io/io_12/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_12/original/ -o relevance_java_dot/commons-io/io_12/original 
cp relevance_java_bytecode/commons-io/io_12/mutants_info.json relevance_java_dot/commons-io/io_12/ 
cp relevance_java_bytecode/commons-io/io_12/interaction_pairs.csv relevance_java_dot/commons-io/io_12/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_24/fom_export/ -o relevance_java_dot/commons-io/io_24/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_24/original/ -o relevance_java_dot/commons-io/io_24/original 
cp relevance_java_bytecode/commons-io/io_24/mutants_info.json relevance_java_dot/commons-io/io_24/ 
cp relevance_java_bytecode/commons-io/io_24/interaction_pairs.csv relevance_java_dot/commons-io/io_24/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_17/fom_export/ -o relevance_java_dot/commons-io/io_17/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_17/original/ -o relevance_java_dot/commons-io/io_17/original 
cp relevance_java_bytecode/commons-io/io_17/mutants_info.json relevance_java_dot/commons-io/io_17/ 
cp relevance_java_bytecode/commons-io/io_17/interaction_pairs.csv relevance_java_dot/commons-io/io_17/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_18/fom_export/ -o relevance_java_dot/commons-io/io_18/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_18/original/ -o relevance_java_dot/commons-io/io_18/original 
cp relevance_java_bytecode/commons-io/io_18/mutants_info.json relevance_java_dot/commons-io/io_18/ 
cp relevance_java_bytecode/commons-io/io_18/interaction_pairs.csv relevance_java_dot/commons-io/io_18/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_22/fom_export/ -o relevance_java_dot/commons-io/io_22/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_22/original/ -o relevance_java_dot/commons-io/io_22/original 
cp relevance_java_bytecode/commons-io/io_22/mutants_info.json relevance_java_dot/commons-io/io_22/ 
cp relevance_java_bytecode/commons-io/io_22/interaction_pairs.csv relevance_java_dot/commons-io/io_22/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_13/fom_export/ -o relevance_java_dot/commons-io/io_13/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_13/original/ -o relevance_java_dot/commons-io/io_13/original 
cp relevance_java_bytecode/commons-io/io_13/mutants_info.json relevance_java_dot/commons-io/io_13/ 
cp relevance_java_bytecode/commons-io/io_13/interaction_pairs.csv relevance_java_dot/commons-io/io_13/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_8/fom_export/ -o relevance_java_dot/commons-io/io_8/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_8/original/ -o relevance_java_dot/commons-io/io_8/original 
cp relevance_java_bytecode/commons-io/io_8/mutants_info.json relevance_java_dot/commons-io/io_8/ 
cp relevance_java_bytecode/commons-io/io_8/interaction_pairs.csv relevance_java_dot/commons-io/io_8/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_10/fom_export/ -o relevance_java_dot/commons-io/io_10/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_10/original/ -o relevance_java_dot/commons-io/io_10/original 
cp relevance_java_bytecode/commons-io/io_10/mutants_info.json relevance_java_dot/commons-io/io_10/ 
cp relevance_java_bytecode/commons-io/io_10/interaction_pairs.csv relevance_java_dot/commons-io/io_10/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_5/fom_export/ -o relevance_java_dot/commons-io/io_5/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_5/original/ -o relevance_java_dot/commons-io/io_5/original 
cp relevance_java_bytecode/commons-io/io_5/mutants_info.json relevance_java_dot/commons-io/io_5/ 
cp relevance_java_bytecode/commons-io/io_5/interaction_pairs.csv relevance_java_dot/commons-io/io_5/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_2/fom_export/ -o relevance_java_dot/commons-io/io_2/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_2/original/ -o relevance_java_dot/commons-io/io_2/original 
cp relevance_java_bytecode/commons-io/io_2/mutants_info.json relevance_java_dot/commons-io/io_2/ 
cp relevance_java_bytecode/commons-io/io_2/interaction_pairs.csv relevance_java_dot/commons-io/io_2/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_19/fom_export/ -o relevance_java_dot/commons-io/io_19/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_19/original/ -o relevance_java_dot/commons-io/io_19/original 
cp relevance_java_bytecode/commons-io/io_19/mutants_info.json relevance_java_dot/commons-io/io_19/ 
cp relevance_java_bytecode/commons-io/io_19/interaction_pairs.csv relevance_java_dot/commons-io/io_19/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_20/fom_export/ -o relevance_java_dot/commons-io/io_20/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_20/original/ -o relevance_java_dot/commons-io/io_20/original 
cp relevance_java_bytecode/commons-io/io_20/mutants_info.json relevance_java_dot/commons-io/io_20/ 
cp relevance_java_bytecode/commons-io/io_20/interaction_pairs.csv relevance_java_dot/commons-io/io_20/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_16/fom_export/ -o relevance_java_dot/commons-io/io_16/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_16/original/ -o relevance_java_dot/commons-io/io_16/original 
cp relevance_java_bytecode/commons-io/io_16/mutants_info.json relevance_java_dot/commons-io/io_16/ 
cp relevance_java_bytecode/commons-io/io_16/interaction_pairs.csv relevance_java_dot/commons-io/io_16/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_4/fom_export/ -o relevance_java_dot/commons-io/io_4/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_4/original/ -o relevance_java_dot/commons-io/io_4/original 
cp relevance_java_bytecode/commons-io/io_4/mutants_info.json relevance_java_dot/commons-io/io_4/ 
cp relevance_java_bytecode/commons-io/io_4/interaction_pairs.csv relevance_java_dot/commons-io/io_4/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_25/fom_export/ -o relevance_java_dot/commons-io/io_25/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_25/original/ -o relevance_java_dot/commons-io/io_25/original 
cp relevance_java_bytecode/commons-io/io_25/mutants_info.json relevance_java_dot/commons-io/io_25/ 
cp relevance_java_bytecode/commons-io/io_25/interaction_pairs.csv relevance_java_dot/commons-io/io_25/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_7/fom_export/ -o relevance_java_dot/commons-io/io_7/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_7/original/ -o relevance_java_dot/commons-io/io_7/original 
cp relevance_java_bytecode/commons-io/io_7/mutants_info.json relevance_java_dot/commons-io/io_7/ 
cp relevance_java_bytecode/commons-io/io_7/interaction_pairs.csv relevance_java_dot/commons-io/io_7/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_14/fom_export/ -o relevance_java_dot/commons-io/io_14/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_14/original/ -o relevance_java_dot/commons-io/io_14/original 
cp relevance_java_bytecode/commons-io/io_14/mutants_info.json relevance_java_dot/commons-io/io_14/ 
cp relevance_java_bytecode/commons-io/io_14/interaction_pairs.csv relevance_java_dot/commons-io/io_14/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_11/fom_export/ -o relevance_java_dot/commons-io/io_11/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_11/original/ -o relevance_java_dot/commons-io/io_11/original 
cp relevance_java_bytecode/commons-io/io_11/mutants_info.json relevance_java_dot/commons-io/io_11/ 
cp relevance_java_bytecode/commons-io/io_11/interaction_pairs.csv relevance_java_dot/commons-io/io_11/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_26/fom_export/ -o relevance_java_dot/commons-io/io_26/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_26/original/ -o relevance_java_dot/commons-io/io_26/original 
cp relevance_java_bytecode/commons-io/io_26/mutants_info.json relevance_java_dot/commons-io/io_26/ 
cp relevance_java_bytecode/commons-io/io_26/interaction_pairs.csv relevance_java_dot/commons-io/io_26/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_3/fom_export/ -o relevance_java_dot/commons-io/io_3/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_3/original/ -o relevance_java_dot/commons-io/io_3/original 
cp relevance_java_bytecode/commons-io/io_3/mutants_info.json relevance_java_dot/commons-io/io_3/ 
cp relevance_java_bytecode/commons-io/io_3/interaction_pairs.csv relevance_java_dot/commons-io/io_3/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_15/fom_export/ -o relevance_java_dot/commons-io/io_15/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-io/io_15/original/ -o relevance_java_dot/commons-io/io_15/original 
cp relevance_java_bytecode/commons-io/io_15/mutants_info.json relevance_java_dot/commons-io/io_15/ 
cp relevance_java_bytecode/commons-io/io_15/interaction_pairs.csv relevance_java_dot/commons-io/io_15/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_8/fom_export/ -o relevance_java_dot/commons-collections/collections_8/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_8/original/ -o relevance_java_dot/commons-collections/collections_8/original 
cp relevance_java_bytecode/commons-collections/collections_8/mutants_info.json relevance_java_dot/commons-collections/collections_8/ 
cp relevance_java_bytecode/commons-collections/collections_8/interaction_pairs.csv relevance_java_dot/commons-collections/collections_8/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_19/fom_export/ -o relevance_java_dot/commons-collections/collections_19/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_19/original/ -o relevance_java_dot/commons-collections/collections_19/original 
cp relevance_java_bytecode/commons-collections/collections_19/mutants_info.json relevance_java_dot/commons-collections/collections_19/ 
cp relevance_java_bytecode/commons-collections/collections_19/interaction_pairs.csv relevance_java_dot/commons-collections/collections_19/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_25/fom_export/ -o relevance_java_dot/commons-collections/collections_25/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_25/original/ -o relevance_java_dot/commons-collections/collections_25/original 
cp relevance_java_bytecode/commons-collections/collections_25/mutants_info.json relevance_java_dot/commons-collections/collections_25/ 
cp relevance_java_bytecode/commons-collections/collections_25/interaction_pairs.csv relevance_java_dot/commons-collections/collections_25/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_28/fom_export/ -o relevance_java_dot/commons-collections/collections_28/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_28/original/ -o relevance_java_dot/commons-collections/collections_28/original 
cp relevance_java_bytecode/commons-collections/collections_28/mutants_info.json relevance_java_dot/commons-collections/collections_28/ 
cp relevance_java_bytecode/commons-collections/collections_28/interaction_pairs.csv relevance_java_dot/commons-collections/collections_28/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_10/fom_export/ -o relevance_java_dot/commons-collections/collections_10/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_10/original/ -o relevance_java_dot/commons-collections/collections_10/original 
cp relevance_java_bytecode/commons-collections/collections_10/mutants_info.json relevance_java_dot/commons-collections/collections_10/ 
cp relevance_java_bytecode/commons-collections/collections_10/interaction_pairs.csv relevance_java_dot/commons-collections/collections_10/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_26/fom_export/ -o relevance_java_dot/commons-collections/collections_26/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_26/original/ -o relevance_java_dot/commons-collections/collections_26/original 
cp relevance_java_bytecode/commons-collections/collections_26/mutants_info.json relevance_java_dot/commons-collections/collections_26/ 
cp relevance_java_bytecode/commons-collections/collections_26/interaction_pairs.csv relevance_java_dot/commons-collections/collections_26/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_4/fom_export/ -o relevance_java_dot/commons-collections/collections_4/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_4/original/ -o relevance_java_dot/commons-collections/collections_4/original 
cp relevance_java_bytecode/commons-collections/collections_4/mutants_info.json relevance_java_dot/commons-collections/collections_4/ 
cp relevance_java_bytecode/commons-collections/collections_4/interaction_pairs.csv relevance_java_dot/commons-collections/collections_4/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_20/fom_export/ -o relevance_java_dot/commons-collections/collections_20/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_20/original/ -o relevance_java_dot/commons-collections/collections_20/original 
cp relevance_java_bytecode/commons-collections/collections_20/mutants_info.json relevance_java_dot/commons-collections/collections_20/ 
cp relevance_java_bytecode/commons-collections/collections_20/interaction_pairs.csv relevance_java_dot/commons-collections/collections_20/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_14/fom_export/ -o relevance_java_dot/commons-collections/collections_14/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_14/original/ -o relevance_java_dot/commons-collections/collections_14/original 
cp relevance_java_bytecode/commons-collections/collections_14/mutants_info.json relevance_java_dot/commons-collections/collections_14/ 
cp relevance_java_bytecode/commons-collections/collections_14/interaction_pairs.csv relevance_java_dot/commons-collections/collections_14/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_15/fom_export/ -o relevance_java_dot/commons-collections/collections_15/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_15/original/ -o relevance_java_dot/commons-collections/collections_15/original 
cp relevance_java_bytecode/commons-collections/collections_15/mutants_info.json relevance_java_dot/commons-collections/collections_15/ 
cp relevance_java_bytecode/commons-collections/collections_15/interaction_pairs.csv relevance_java_dot/commons-collections/collections_15/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_21/fom_export/ -o relevance_java_dot/commons-collections/collections_21/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_21/original/ -o relevance_java_dot/commons-collections/collections_21/original 
cp relevance_java_bytecode/commons-collections/collections_21/mutants_info.json relevance_java_dot/commons-collections/collections_21/ 
cp relevance_java_bytecode/commons-collections/collections_21/interaction_pairs.csv relevance_java_dot/commons-collections/collections_21/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_27/fom_export/ -o relevance_java_dot/commons-collections/collections_27/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_27/original/ -o relevance_java_dot/commons-collections/collections_27/original 
cp relevance_java_bytecode/commons-collections/collections_27/mutants_info.json relevance_java_dot/commons-collections/collections_27/ 
cp relevance_java_bytecode/commons-collections/collections_27/interaction_pairs.csv relevance_java_dot/commons-collections/collections_27/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_9/fom_export/ -o relevance_java_dot/commons-collections/collections_9/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_9/original/ -o relevance_java_dot/commons-collections/collections_9/original 
cp relevance_java_bytecode/commons-collections/collections_9/mutants_info.json relevance_java_dot/commons-collections/collections_9/ 
cp relevance_java_bytecode/commons-collections/collections_9/interaction_pairs.csv relevance_java_dot/commons-collections/collections_9/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_13/fom_export/ -o relevance_java_dot/commons-collections/collections_13/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_13/original/ -o relevance_java_dot/commons-collections/collections_13/original 
cp relevance_java_bytecode/commons-collections/collections_13/mutants_info.json relevance_java_dot/commons-collections/collections_13/ 
cp relevance_java_bytecode/commons-collections/collections_13/interaction_pairs.csv relevance_java_dot/commons-collections/collections_13/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_1/fom_export/ -o relevance_java_dot/commons-collections/collections_1/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_1/original/ -o relevance_java_dot/commons-collections/collections_1/original 
cp relevance_java_bytecode/commons-collections/collections_1/mutants_info.json relevance_java_dot/commons-collections/collections_1/ 
cp relevance_java_bytecode/commons-collections/collections_1/interaction_pairs.csv relevance_java_dot/commons-collections/collections_1/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_11/fom_export/ -o relevance_java_dot/commons-collections/collections_11/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_11/original/ -o relevance_java_dot/commons-collections/collections_11/original 
cp relevance_java_bytecode/commons-collections/collections_11/mutants_info.json relevance_java_dot/commons-collections/collections_11/ 
cp relevance_java_bytecode/commons-collections/collections_11/interaction_pairs.csv relevance_java_dot/commons-collections/collections_11/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_3/fom_export/ -o relevance_java_dot/commons-collections/collections_3/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_3/original/ -o relevance_java_dot/commons-collections/collections_3/original 
cp relevance_java_bytecode/commons-collections/collections_3/mutants_info.json relevance_java_dot/commons-collections/collections_3/ 
cp relevance_java_bytecode/commons-collections/collections_3/interaction_pairs.csv relevance_java_dot/commons-collections/collections_3/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_17/fom_export/ -o relevance_java_dot/commons-collections/collections_17/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_17/original/ -o relevance_java_dot/commons-collections/collections_17/original 
cp relevance_java_bytecode/commons-collections/collections_17/mutants_info.json relevance_java_dot/commons-collections/collections_17/ 
cp relevance_java_bytecode/commons-collections/collections_17/interaction_pairs.csv relevance_java_dot/commons-collections/collections_17/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_5/fom_export/ -o relevance_java_dot/commons-collections/collections_5/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_5/original/ -o relevance_java_dot/commons-collections/collections_5/original 
cp relevance_java_bytecode/commons-collections/collections_5/mutants_info.json relevance_java_dot/commons-collections/collections_5/ 
cp relevance_java_bytecode/commons-collections/collections_5/interaction_pairs.csv relevance_java_dot/commons-collections/collections_5/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_18/fom_export/ -o relevance_java_dot/commons-collections/collections_18/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_18/original/ -o relevance_java_dot/commons-collections/collections_18/original 
cp relevance_java_bytecode/commons-collections/collections_18/mutants_info.json relevance_java_dot/commons-collections/collections_18/ 
cp relevance_java_bytecode/commons-collections/collections_18/interaction_pairs.csv relevance_java_dot/commons-collections/collections_18/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_22/fom_export/ -o relevance_java_dot/commons-collections/collections_22/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_22/original/ -o relevance_java_dot/commons-collections/collections_22/original 
cp relevance_java_bytecode/commons-collections/collections_22/mutants_info.json relevance_java_dot/commons-collections/collections_22/ 
cp relevance_java_bytecode/commons-collections/collections_22/interaction_pairs.csv relevance_java_dot/commons-collections/collections_22/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_12/fom_export/ -o relevance_java_dot/commons-collections/collections_12/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_12/original/ -o relevance_java_dot/commons-collections/collections_12/original 
cp relevance_java_bytecode/commons-collections/collections_12/mutants_info.json relevance_java_dot/commons-collections/collections_12/ 
cp relevance_java_bytecode/commons-collections/collections_12/interaction_pairs.csv relevance_java_dot/commons-collections/collections_12/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_7/fom_export/ -o relevance_java_dot/commons-collections/collections_7/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-collections/collections_7/original/ -o relevance_java_dot/commons-collections/collections_7/original 
cp relevance_java_bytecode/commons-collections/collections_7/mutants_info.json relevance_java_dot/commons-collections/collections_7/ 
cp relevance_java_bytecode/commons-collections/collections_7/interaction_pairs.csv relevance_java_dot/commons-collections/collections_7/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_30/fom_export/ -o relevance_java_dot/commons-text/text_30/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_30/original/ -o relevance_java_dot/commons-text/text_30/original 
cp relevance_java_bytecode/commons-text/text_30/mutants_info.json relevance_java_dot/commons-text/text_30/ 
cp relevance_java_bytecode/commons-text/text_30/interaction_pairs.csv relevance_java_dot/commons-text/text_30/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_5/fom_export/ -o relevance_java_dot/commons-text/text_5/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_5/original/ -o relevance_java_dot/commons-text/text_5/original 
cp relevance_java_bytecode/commons-text/text_5/mutants_info.json relevance_java_dot/commons-text/text_5/ 
cp relevance_java_bytecode/commons-text/text_5/interaction_pairs.csv relevance_java_dot/commons-text/text_5/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_31/fom_export/ -o relevance_java_dot/commons-text/text_31/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_31/original/ -o relevance_java_dot/commons-text/text_31/original 
cp relevance_java_bytecode/commons-text/text_31/mutants_info.json relevance_java_dot/commons-text/text_31/ 
cp relevance_java_bytecode/commons-text/text_31/interaction_pairs.csv relevance_java_dot/commons-text/text_31/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_14/fom_export/ -o relevance_java_dot/commons-text/text_14/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_14/original/ -o relevance_java_dot/commons-text/text_14/original 
cp relevance_java_bytecode/commons-text/text_14/mutants_info.json relevance_java_dot/commons-text/text_14/ 
cp relevance_java_bytecode/commons-text/text_14/interaction_pairs.csv relevance_java_dot/commons-text/text_14/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_24/fom_export/ -o relevance_java_dot/commons-text/text_24/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_24/original/ -o relevance_java_dot/commons-text/text_24/original 
cp relevance_java_bytecode/commons-text/text_24/mutants_info.json relevance_java_dot/commons-text/text_24/ 
cp relevance_java_bytecode/commons-text/text_24/interaction_pairs.csv relevance_java_dot/commons-text/text_24/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_29/fom_export/ -o relevance_java_dot/commons-text/text_29/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_29/original/ -o relevance_java_dot/commons-text/text_29/original 
cp relevance_java_bytecode/commons-text/text_29/mutants_info.json relevance_java_dot/commons-text/text_29/ 
cp relevance_java_bytecode/commons-text/text_29/interaction_pairs.csv relevance_java_dot/commons-text/text_29/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_12/fom_export/ -o relevance_java_dot/commons-text/text_12/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_12/original/ -o relevance_java_dot/commons-text/text_12/original 
cp relevance_java_bytecode/commons-text/text_12/mutants_info.json relevance_java_dot/commons-text/text_12/ 
cp relevance_java_bytecode/commons-text/text_12/interaction_pairs.csv relevance_java_dot/commons-text/text_12/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_16/fom_export/ -o relevance_java_dot/commons-text/text_16/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_16/original/ -o relevance_java_dot/commons-text/text_16/original 
cp relevance_java_bytecode/commons-text/text_16/mutants_info.json relevance_java_dot/commons-text/text_16/ 
cp relevance_java_bytecode/commons-text/text_16/interaction_pairs.csv relevance_java_dot/commons-text/text_16/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_7/fom_export/ -o relevance_java_dot/commons-text/text_7/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_7/original/ -o relevance_java_dot/commons-text/text_7/original 
cp relevance_java_bytecode/commons-text/text_7/mutants_info.json relevance_java_dot/commons-text/text_7/ 
cp relevance_java_bytecode/commons-text/text_7/interaction_pairs.csv relevance_java_dot/commons-text/text_7/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_4/fom_export/ -o relevance_java_dot/commons-text/text_4/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_4/original/ -o relevance_java_dot/commons-text/text_4/original 
cp relevance_java_bytecode/commons-text/text_4/mutants_info.json relevance_java_dot/commons-text/text_4/ 
cp relevance_java_bytecode/commons-text/text_4/interaction_pairs.csv relevance_java_dot/commons-text/text_4/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_6/fom_export/ -o relevance_java_dot/commons-text/text_6/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_6/original/ -o relevance_java_dot/commons-text/text_6/original 
cp relevance_java_bytecode/commons-text/text_6/mutants_info.json relevance_java_dot/commons-text/text_6/ 
cp relevance_java_bytecode/commons-text/text_6/interaction_pairs.csv relevance_java_dot/commons-text/text_6/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_23/fom_export/ -o relevance_java_dot/commons-text/text_23/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_23/original/ -o relevance_java_dot/commons-text/text_23/original 
cp relevance_java_bytecode/commons-text/text_23/mutants_info.json relevance_java_dot/commons-text/text_23/ 
cp relevance_java_bytecode/commons-text/text_23/interaction_pairs.csv relevance_java_dot/commons-text/text_23/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_2/fom_export/ -o relevance_java_dot/commons-text/text_2/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_2/original/ -o relevance_java_dot/commons-text/text_2/original 
cp relevance_java_bytecode/commons-text/text_2/mutants_info.json relevance_java_dot/commons-text/text_2/ 
cp relevance_java_bytecode/commons-text/text_2/interaction_pairs.csv relevance_java_dot/commons-text/text_2/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_28/fom_export/ -o relevance_java_dot/commons-text/text_28/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_28/original/ -o relevance_java_dot/commons-text/text_28/original 
cp relevance_java_bytecode/commons-text/text_28/mutants_info.json relevance_java_dot/commons-text/text_28/ 
cp relevance_java_bytecode/commons-text/text_28/interaction_pairs.csv relevance_java_dot/commons-text/text_28/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_25/fom_export/ -o relevance_java_dot/commons-text/text_25/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_25/original/ -o relevance_java_dot/commons-text/text_25/original 
cp relevance_java_bytecode/commons-text/text_25/mutants_info.json relevance_java_dot/commons-text/text_25/ 
cp relevance_java_bytecode/commons-text/text_25/interaction_pairs.csv relevance_java_dot/commons-text/text_25/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_22/fom_export/ -o relevance_java_dot/commons-text/text_22/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_22/original/ -o relevance_java_dot/commons-text/text_22/original 
cp relevance_java_bytecode/commons-text/text_22/mutants_info.json relevance_java_dot/commons-text/text_22/ 
cp relevance_java_bytecode/commons-text/text_22/interaction_pairs.csv relevance_java_dot/commons-text/text_22/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_1/fom_export/ -o relevance_java_dot/commons-text/text_1/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_1/original/ -o relevance_java_dot/commons-text/text_1/original 
cp relevance_java_bytecode/commons-text/text_1/mutants_info.json relevance_java_dot/commons-text/text_1/ 
cp relevance_java_bytecode/commons-text/text_1/interaction_pairs.csv relevance_java_dot/commons-text/text_1/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_11/fom_export/ -o relevance_java_dot/commons-text/text_11/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_11/original/ -o relevance_java_dot/commons-text/text_11/original 
cp relevance_java_bytecode/commons-text/text_11/mutants_info.json relevance_java_dot/commons-text/text_11/ 
cp relevance_java_bytecode/commons-text/text_11/interaction_pairs.csv relevance_java_dot/commons-text/text_11/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_3/fom_export/ -o relevance_java_dot/commons-text/text_3/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_3/original/ -o relevance_java_dot/commons-text/text_3/original 
cp relevance_java_bytecode/commons-text/text_3/mutants_info.json relevance_java_dot/commons-text/text_3/ 
cp relevance_java_bytecode/commons-text/text_3/interaction_pairs.csv relevance_java_dot/commons-text/text_3/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_20/fom_export/ -o relevance_java_dot/commons-text/text_20/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_20/original/ -o relevance_java_dot/commons-text/text_20/original 
cp relevance_java_bytecode/commons-text/text_20/mutants_info.json relevance_java_dot/commons-text/text_20/ 
cp relevance_java_bytecode/commons-text/text_20/interaction_pairs.csv relevance_java_dot/commons-text/text_20/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_26/fom_export/ -o relevance_java_dot/commons-text/text_26/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_26/original/ -o relevance_java_dot/commons-text/text_26/original 
cp relevance_java_bytecode/commons-text/text_26/mutants_info.json relevance_java_dot/commons-text/text_26/ 
cp relevance_java_bytecode/commons-text/text_26/interaction_pairs.csv relevance_java_dot/commons-text/text_26/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_13/fom_export/ -o relevance_java_dot/commons-text/text_13/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_13/original/ -o relevance_java_dot/commons-text/text_13/original 
cp relevance_java_bytecode/commons-text/text_13/mutants_info.json relevance_java_dot/commons-text/text_13/ 
cp relevance_java_bytecode/commons-text/text_13/interaction_pairs.csv relevance_java_dot/commons-text/text_13/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_15/fom_export/ -o relevance_java_dot/commons-text/text_15/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_15/original/ -o relevance_java_dot/commons-text/text_15/original 
cp relevance_java_bytecode/commons-text/text_15/mutants_info.json relevance_java_dot/commons-text/text_15/ 
cp relevance_java_bytecode/commons-text/text_15/interaction_pairs.csv relevance_java_dot/commons-text/text_15/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_19/fom_export/ -o relevance_java_dot/commons-text/text_19/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_19/original/ -o relevance_java_dot/commons-text/text_19/original 
cp relevance_java_bytecode/commons-text/text_19/mutants_info.json relevance_java_dot/commons-text/text_19/ 
cp relevance_java_bytecode/commons-text/text_19/interaction_pairs.csv relevance_java_dot/commons-text/text_19/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_8/fom_export/ -o relevance_java_dot/commons-text/text_8/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_8/original/ -o relevance_java_dot/commons-text/text_8/original 
cp relevance_java_bytecode/commons-text/text_8/mutants_info.json relevance_java_dot/commons-text/text_8/ 
cp relevance_java_bytecode/commons-text/text_8/interaction_pairs.csv relevance_java_dot/commons-text/text_8/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_21/fom_export/ -o relevance_java_dot/commons-text/text_21/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_21/original/ -o relevance_java_dot/commons-text/text_21/original 
cp relevance_java_bytecode/commons-text/text_21/mutants_info.json relevance_java_dot/commons-text/text_21/ 
cp relevance_java_bytecode/commons-text/text_21/interaction_pairs.csv relevance_java_dot/commons-text/text_21/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_18/fom_export/ -o relevance_java_dot/commons-text/text_18/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_18/original/ -o relevance_java_dot/commons-text/text_18/original 
cp relevance_java_bytecode/commons-text/text_18/mutants_info.json relevance_java_dot/commons-text/text_18/ 
cp relevance_java_bytecode/commons-text/text_18/interaction_pairs.csv relevance_java_dot/commons-text/text_18/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_9/fom_export/ -o relevance_java_dot/commons-text/text_9/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-text/text_9/original/ -o relevance_java_dot/commons-text/text_9/original 
cp relevance_java_bytecode/commons-text/text_9/mutants_info.json relevance_java_dot/commons-text/text_9/ 
cp relevance_java_bytecode/commons-text/text_9/interaction_pairs.csv relevance_java_dot/commons-text/text_9/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_18/fom_export/ -o relevance_java_dot/commons-csv/csv_18/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_18/original/ -o relevance_java_dot/commons-csv/csv_18/original 
cp relevance_java_bytecode/commons-csv/csv_18/mutants_info.json relevance_java_dot/commons-csv/csv_18/ 
cp relevance_java_bytecode/commons-csv/csv_18/interaction_pairs.csv relevance_java_dot/commons-csv/csv_18/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_22/fom_export/ -o relevance_java_dot/commons-csv/csv_22/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_22/original/ -o relevance_java_dot/commons-csv/csv_22/original 
cp relevance_java_bytecode/commons-csv/csv_22/mutants_info.json relevance_java_dot/commons-csv/csv_22/ 
cp relevance_java_bytecode/commons-csv/csv_22/interaction_pairs.csv relevance_java_dot/commons-csv/csv_22/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_27/fom_export/ -o relevance_java_dot/commons-csv/csv_27/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_27/original/ -o relevance_java_dot/commons-csv/csv_27/original 
cp relevance_java_bytecode/commons-csv/csv_27/mutants_info.json relevance_java_dot/commons-csv/csv_27/ 
cp relevance_java_bytecode/commons-csv/csv_27/interaction_pairs.csv relevance_java_dot/commons-csv/csv_27/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_2/fom_export/ -o relevance_java_dot/commons-csv/csv_2/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_2/original/ -o relevance_java_dot/commons-csv/csv_2/original 
cp relevance_java_bytecode/commons-csv/csv_2/mutants_info.json relevance_java_dot/commons-csv/csv_2/ 
cp relevance_java_bytecode/commons-csv/csv_2/interaction_pairs.csv relevance_java_dot/commons-csv/csv_2/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_6/fom_export/ -o relevance_java_dot/commons-csv/csv_6/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_6/original/ -o relevance_java_dot/commons-csv/csv_6/original 
cp relevance_java_bytecode/commons-csv/csv_6/mutants_info.json relevance_java_dot/commons-csv/csv_6/ 
cp relevance_java_bytecode/commons-csv/csv_6/interaction_pairs.csv relevance_java_dot/commons-csv/csv_6/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_16/fom_export/ -o relevance_java_dot/commons-csv/csv_16/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_16/original/ -o relevance_java_dot/commons-csv/csv_16/original 
cp relevance_java_bytecode/commons-csv/csv_16/mutants_info.json relevance_java_dot/commons-csv/csv_16/ 
cp relevance_java_bytecode/commons-csv/csv_16/interaction_pairs.csv relevance_java_dot/commons-csv/csv_16/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_14/fom_export/ -o relevance_java_dot/commons-csv/csv_14/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_14/original/ -o relevance_java_dot/commons-csv/csv_14/original 
cp relevance_java_bytecode/commons-csv/csv_14/mutants_info.json relevance_java_dot/commons-csv/csv_14/ 
cp relevance_java_bytecode/commons-csv/csv_14/interaction_pairs.csv relevance_java_dot/commons-csv/csv_14/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_1/fom_export/ -o relevance_java_dot/commons-csv/csv_1/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_1/original/ -o relevance_java_dot/commons-csv/csv_1/original 
cp relevance_java_bytecode/commons-csv/csv_1/mutants_info.json relevance_java_dot/commons-csv/csv_1/ 
cp relevance_java_bytecode/commons-csv/csv_1/interaction_pairs.csv relevance_java_dot/commons-csv/csv_1/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_8/fom_export/ -o relevance_java_dot/commons-csv/csv_8/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_8/original/ -o relevance_java_dot/commons-csv/csv_8/original 
cp relevance_java_bytecode/commons-csv/csv_8/mutants_info.json relevance_java_dot/commons-csv/csv_8/ 
cp relevance_java_bytecode/commons-csv/csv_8/interaction_pairs.csv relevance_java_dot/commons-csv/csv_8/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_13/fom_export/ -o relevance_java_dot/commons-csv/csv_13/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_13/original/ -o relevance_java_dot/commons-csv/csv_13/original 
cp relevance_java_bytecode/commons-csv/csv_13/mutants_info.json relevance_java_dot/commons-csv/csv_13/ 
cp relevance_java_bytecode/commons-csv/csv_13/interaction_pairs.csv relevance_java_dot/commons-csv/csv_13/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_20/fom_export/ -o relevance_java_dot/commons-csv/csv_20/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_20/original/ -o relevance_java_dot/commons-csv/csv_20/original 
cp relevance_java_bytecode/commons-csv/csv_20/mutants_info.json relevance_java_dot/commons-csv/csv_20/ 
cp relevance_java_bytecode/commons-csv/csv_20/interaction_pairs.csv relevance_java_dot/commons-csv/csv_20/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_29/fom_export/ -o relevance_java_dot/commons-csv/csv_29/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_29/original/ -o relevance_java_dot/commons-csv/csv_29/original 
cp relevance_java_bytecode/commons-csv/csv_29/mutants_info.json relevance_java_dot/commons-csv/csv_29/ 
cp relevance_java_bytecode/commons-csv/csv_29/interaction_pairs.csv relevance_java_dot/commons-csv/csv_29/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_21/fom_export/ -o relevance_java_dot/commons-csv/csv_21/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_21/original/ -o relevance_java_dot/commons-csv/csv_21/original 
cp relevance_java_bytecode/commons-csv/csv_21/mutants_info.json relevance_java_dot/commons-csv/csv_21/ 
cp relevance_java_bytecode/commons-csv/csv_21/interaction_pairs.csv relevance_java_dot/commons-csv/csv_21/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_10/fom_export/ -o relevance_java_dot/commons-csv/csv_10/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_10/original/ -o relevance_java_dot/commons-csv/csv_10/original 
cp relevance_java_bytecode/commons-csv/csv_10/mutants_info.json relevance_java_dot/commons-csv/csv_10/ 
cp relevance_java_bytecode/commons-csv/csv_10/interaction_pairs.csv relevance_java_dot/commons-csv/csv_10/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_23/fom_export/ -o relevance_java_dot/commons-csv/csv_23/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_23/original/ -o relevance_java_dot/commons-csv/csv_23/original 
cp relevance_java_bytecode/commons-csv/csv_23/mutants_info.json relevance_java_dot/commons-csv/csv_23/ 
cp relevance_java_bytecode/commons-csv/csv_23/interaction_pairs.csv relevance_java_dot/commons-csv/csv_23/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_25/fom_export/ -o relevance_java_dot/commons-csv/csv_25/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_25/original/ -o relevance_java_dot/commons-csv/csv_25/original 
cp relevance_java_bytecode/commons-csv/csv_25/mutants_info.json relevance_java_dot/commons-csv/csv_25/ 
cp relevance_java_bytecode/commons-csv/csv_25/interaction_pairs.csv relevance_java_dot/commons-csv/csv_25/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_12/fom_export/ -o relevance_java_dot/commons-csv/csv_12/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_12/original/ -o relevance_java_dot/commons-csv/csv_12/original 
cp relevance_java_bytecode/commons-csv/csv_12/mutants_info.json relevance_java_dot/commons-csv/csv_12/ 
cp relevance_java_bytecode/commons-csv/csv_12/interaction_pairs.csv relevance_java_dot/commons-csv/csv_12/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_7/fom_export/ -o relevance_java_dot/commons-csv/csv_7/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_7/original/ -o relevance_java_dot/commons-csv/csv_7/original 
cp relevance_java_bytecode/commons-csv/csv_7/mutants_info.json relevance_java_dot/commons-csv/csv_7/ 
cp relevance_java_bytecode/commons-csv/csv_7/interaction_pairs.csv relevance_java_dot/commons-csv/csv_7/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_15/fom_export/ -o relevance_java_dot/commons-csv/csv_15/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_15/original/ -o relevance_java_dot/commons-csv/csv_15/original 
cp relevance_java_bytecode/commons-csv/csv_15/mutants_info.json relevance_java_dot/commons-csv/csv_15/ 
cp relevance_java_bytecode/commons-csv/csv_15/interaction_pairs.csv relevance_java_dot/commons-csv/csv_15/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_4/fom_export/ -o relevance_java_dot/commons-csv/csv_4/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_4/original/ -o relevance_java_dot/commons-csv/csv_4/original 
cp relevance_java_bytecode/commons-csv/csv_4/mutants_info.json relevance_java_dot/commons-csv/csv_4/ 
cp relevance_java_bytecode/commons-csv/csv_4/interaction_pairs.csv relevance_java_dot/commons-csv/csv_4/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_19/fom_export/ -o relevance_java_dot/commons-csv/csv_19/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_19/original/ -o relevance_java_dot/commons-csv/csv_19/original 
cp relevance_java_bytecode/commons-csv/csv_19/mutants_info.json relevance_java_dot/commons-csv/csv_19/ 
cp relevance_java_bytecode/commons-csv/csv_19/interaction_pairs.csv relevance_java_dot/commons-csv/csv_19/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_24/fom_export/ -o relevance_java_dot/commons-csv/csv_24/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_24/original/ -o relevance_java_dot/commons-csv/csv_24/original 
cp relevance_java_bytecode/commons-csv/csv_24/mutants_info.json relevance_java_dot/commons-csv/csv_24/ 
cp relevance_java_bytecode/commons-csv/csv_24/interaction_pairs.csv relevance_java_dot/commons-csv/csv_24/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_5/fom_export/ -o relevance_java_dot/commons-csv/csv_5/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_5/original/ -o relevance_java_dot/commons-csv/csv_5/original 
cp relevance_java_bytecode/commons-csv/csv_5/mutants_info.json relevance_java_dot/commons-csv/csv_5/ 
cp relevance_java_bytecode/commons-csv/csv_5/interaction_pairs.csv relevance_java_dot/commons-csv/csv_5/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_31/fom_export/ -o relevance_java_dot/commons-csv/csv_31/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_31/original/ -o relevance_java_dot/commons-csv/csv_31/original 
cp relevance_java_bytecode/commons-csv/csv_31/mutants_info.json relevance_java_dot/commons-csv/csv_31/ 
cp relevance_java_bytecode/commons-csv/csv_31/interaction_pairs.csv relevance_java_dot/commons-csv/csv_31/ 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_17/fom_export/ -o relevance_java_dot/commons-csv/csv_17/mutants 
java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i relevance_java_bytecode/commons-csv/csv_17/original/ -o relevance_java_dot/commons-csv/csv_17/original 
cp relevance_java_bytecode/commons-csv/csv_17/mutants_info.json relevance_java_dot/commons-csv/csv_17/ 
cp relevance_java_bytecode/commons-csv/csv_17/interaction_pairs.csv relevance_java_dot/commons-csv/csv_17/ 
