What the program does/how to navigate the files:
1. GLMNET.h is the header file that contains all details about which functions are available, and which aren't. Further, the documentation for 
    all the functions can be found in here.
2. GLMNET.cpp is the .cpp file that contains all the implentation logic
3. load.h is the header file that contains the parsing function headers
4. load.cpp contains all the implementation logic for parsing
3. main.cpp is the .cpp file that excecutes the code, and shows an example of how you can use the GLMNET object.



To add/improve on in the future:
1. Add an elastic net penalty to the TORRENT updates to allow for sparse weight vectors
2. Error handling, especially when there are size mismatches (very common!) (this will be a problem if we use the load_data_into function instead of the csv_to_matrix function)
3. The implementation time and space complexities can be improved. (for the load_data_into function)
4. Cross-validation, test-errors, etc. are metrics that are not directly available. You can write custom functions in main.cpp to do 
    the same though. (this goes into utils)
5. Graphing capabilities, would be especially useful when doing cross-valdiation
6. Some optimization can be done on standardize_data to prevent copying.