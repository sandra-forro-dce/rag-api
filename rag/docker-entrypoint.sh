#!/bin/bash

echo "Container is running!!!"

# args="$@"
# echo $args

# if [[ -z ${args} ]]; 
# then
#     pipenv shell
# else
#   pipenv run python rag.py #$args
# fi
# pipenv run python rag.py

pipenv run uvicorn rag:app --host 0.0.0.0 --port 9000
echo "API is now running... "
echo "Use http://localhost:9000/rag/str for string input"
echo "Use http://localhost:9000/rag for json input"
