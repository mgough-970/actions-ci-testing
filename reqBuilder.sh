#!/bin/bash

echo Parsing requirements from ./notebooks
find notebooks/ -name "requirements.txt" -exec sh -c 'x="{}"; mv "$x" "${x}.orig"; git add "${x}.orig"' \;
find ./notebooks -name "*.ipynb" | xargs -I {} pipreqsnb {}
find ./notebooks -name 'requirements.txt' -exec sed -i '' 's/==.*//' {} \;
find ./notebooks -name 'requirements.txt' -exec sed -i '' 's/ .*//' {} \;
find ./notebooks -name "requirements.txt" | xargs -I {} cat {} >> reqList.txt
cat reqList.txt | cut -f1 -d " " | cut -f1 -d"=" | sort -u > reqList2.txt
ureqs=`cat reqList2.txt | wc -l`
echo Done: Requirements file built with $ureqs unique requirements: reqsList2.txt
