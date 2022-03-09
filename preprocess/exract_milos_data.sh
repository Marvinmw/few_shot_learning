tar -xzvf milosdataset/commons-lang_data.tar.gz -C relevance_java_bytecode/
tar -xzvf milosdataset/commons-text_data.tar.gz -C relevance_java_bytecode/
tar -xzvf milosdataset/commons-io_data.tar.gz -C relevance_java_bytecode/
tar -xzvf milosdataset/commons-csv_data.tar.gz -C relevance_java_bytecode/
tar -xzvf milosdataset/commons-collections_data.tar.gz -C relevance_java_bytecode/
tar -xzvf milosdataset/wei_work.tar.gz -C relevance_java_bytecode/
cp relevance_java_bytecode/wei_work/** ./relevance_java_bytecode/ -rf
rm -rf relevance_java_bytecode/wei_work/
