#!/bin/bash

echo Parsing requirements from ./notebooks
find ./notebooks -name "*.ipynb" | xargs -I {} pipreqsnb {} > /dev/null 2>&1
find ./notebooks -name "requirements.txt" | xargs -I {} cat {} >> reqList.txt
cat reqList.txt | cut -f1 -d"=" | sort -u > reqList2.txt
ureqs=`cat reqList2.txt | wc -l`
echo Done: Requirements file built with $ureqs unique requirements: reqsList2.txt
