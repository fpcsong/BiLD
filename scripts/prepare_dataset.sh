#! /bin/bash

wget -P ./benchmarking/datasets https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip
wget -P ./benchmarking/datasets https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip
wget -P ./benchmarking/datasets https://dl.fbaipublicfiles.com/glue/superglue/data/v2/CB.zip
wget -P ./benchmarking/datasets https://dl.fbaipublicfiles.com/glue/superglue/data/v2/COPA.zip
wget -O ./benchmarking/datasets/hellaswag.zip https://github.com/rowanz/hellaswag/archive/refs/heads/master.zip
wget -P ./benchmarking/datasets https://dl.fbaipublicfiles.com/glue/superglue/data/v2/MultiRC.zip
wget -O ./benchmarking/datasets/piqa.zip https://github.com/ybisk/ybisk.github.io/archive/refs/heads/master.zip
wget -P ./benchmarking/datasets https://dl.fbaipublicfiles.com/glue/superglue/data/v2/ReCoRD.zip
wget -P ./benchmarking/datasets https://dl.fbaipublicfiles.com/glue/superglue/data/v2/RTE.zip
wget -P ./benchmarking/datasets https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WiC.zip
wget -P ./benchmarking/datasets https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip
wget -P ./benchmarking/datasets https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WSC.zip

cd ./benchmarking/datasets

unzip -o ARC-V1-Feb2018.zip
unzip -o BoolQ.zip
unzip -o CB.zip
unzip -o COPA.zip
unzip -o hellaswag.zip
unzip -o MultiRC.zip
unzip -o piqa.zip
unzip -o ReCoRD.zip
unzip -o RTE.zip
unzip -o WiC.zip
unzip -o winogrande_1.1.zip
unzip -o WSC.zip

mv ARC-V1-Feb2018-2/ARC-Challenge/ arc-c/
mv ARC-V1-Feb2018-2/ARC-Easy/ arc-e/
mv BoolQ/ boolq/
mv CB/ cb/
mv COPA/ copa/
mv hellaswag-master/data/ hellaswag/
mv MultiRC/ multirc/
mv ybisk.github.io-master/piqa/data piqa/
mv ReCoRD/ record/
mv RTE/ rte/
mv WiC/ wic/
mv winogrande_1.1/ winogrande/
mv WSC/ wsc/

rm *.zip
rm -rf ./__MACOSX
rm -rf ./ARC-V1-Feb2018-2
rm -rf ./hellaswag-master
rm -rf ./ybisk.github.io-master

current_directory=$(pwd)

for dir in */; do
  if [ -d "$dir" ]; then
    train_dir="${current_directory}/${dir}train"
    eval_dir="${current_directory}/${dir}eval"
    
    if [ ! -d "$train_dir" ]; then
      mkdir -p "$train_dir"
    fi
    
    if [ ! -d "$eval_dir" ]; then
      mkdir -p "$eval_dir"
    fi
  fi
done

mv arc-c/ARC-Challenge-Train.jsonl arc-c/train/
mv arc-c/ARC-Challenge-Dev.jsonl arc-c/eval/
find arc-c/ -maxdepth 1 -type f -exec rm -f {} \;

mv arc-e/ARC-Easy-Train.jsonl arc-e/train/
mv arc-e/ARC-Easy-Dev.jsonl arc-e/eval/
find arc-e/ -maxdepth 1 -type f -exec rm -f {} \;

mv boolq/train.jsonl boolq/train/
mv boolq/val.jsonl boolq/eval/
find boolq/ -maxdepth 1 -type f -exec rm -f {} \;

mv cb/train.jsonl cb/train/
mv cb/val.jsonl cb/eval/
find cb/ -maxdepth 1 -type f -exec rm -f {} \;

mv copa/train.jsonl copa/train/
mv copa/val.jsonl copa/eval/
find copa/ -maxdepth 1 -type f -exec rm -f {} \;

mv hellaswag/hellaswag_train.jsonl hellaswag/train/
mv hellaswag/hellaswag_val.jsonl hellaswag/eval/
find hellaswag/ -maxdepth 1 -type f -exec rm -f {} \;

mv multirc/train.jsonl multirc/train/
mv multirc/val.jsonl multirc/eval/
find multirc/ -maxdepth 1 -type f -exec rm -f {} \;

mv piqa/train.jsonl piqa/train/
mv piqa/train-labels.lst piqa/train/
mv piqa/valid.jsonl piqa/eval/
mv piqa/valid-labels.lst piqa/eval/
find piqa/ -maxdepth 1 -type f -exec rm -f {} \;

mv record/train.jsonl record/train/
mv record/val.jsonl record/eval/
find record/ -maxdepth 1 -type f -exec rm -f {} \;

mv rte/train.jsonl rte/train/
mv rte/val.jsonl rte/eval/
find rte/ -maxdepth 1 -type f -exec rm -f {} \;

mv wic/train.jsonl wic/train/
mv wic/val.jsonl wic/eval/
find wic/ -maxdepth 1 -type f -exec rm -f {} \;

mv winogrande/train_xl.jsonl winogrande/train/
mv winogrande/train_xl-labels.lst winogrande/train/
mv winogrande/dev.jsonl winogrande/eval/
mv winogrande/dev-labels.lst winogrande/eval/
find winogrande/ -maxdepth 1 -type f -exec rm -f {} \;

mv wsc/train.jsonl wsc/train/
mv wsc/val.jsonl wsc/eval/
find wsc/ -maxdepth 1 -type f -exec rm -f {} \;

cd ../..

python3 ./benchmarking/reformat.py
