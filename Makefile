# Initialize the project
init:
	conda env create -f environment.yml --prefix env;
# Update an already initialized project
upgrade:
	conda env update -f environment.yml --prefix env --prune;
 
# Teardown an initialized project
teardown:
	conda remove --prefix env --all;
