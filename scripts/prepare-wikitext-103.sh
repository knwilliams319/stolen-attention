#!/bin/bash
# NOTE: This script is meant to be called from inside the scripts directory. 
# NOTE: This was adapted from the link below.
# LINK: https://github.com/facebookresearch/fairseq/blob/main/examples/language_model/prepare-wikitext-103.sh

URLS=(
    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"
)
FILES=(
    "wikitext-103-v1.zip"
)

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
            rm $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
            rm $file
        elif [ ${file: -4} == ".zip" ]; then
            unzip $file
            rm $file
        fi
    fi
done

mkdir -p ../data
cp -r wikitext-103 ../data/wikitext-103
rm -r wikitext-103

cd ..
