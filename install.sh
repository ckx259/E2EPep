#!/bin/bash

cd premodel
chmod 777 ./download.sh
bash ./download.sh

cd ..
cd uniprot
tar -xzvf ./human/*.tar.gz -C ./
rm -rf ./human