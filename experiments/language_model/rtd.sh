#!/bin/bash
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
cd $SCRIPT_DIR

cache_dir=/tmp/DeBERTa/RTD/

max_seq_length=512
data_dir=$cache_dir/wiki103/spm_$max_seq_length

function setup_wiki_data(){
	mkdir -p $cache_dir
	if [[ ! -e  $cache_dir/spm.model ]]; then
		wget -q https://huggingface.co/microsoft/deberta-v3-base/resolve/main/spm.model -O $cache_dir/spm.model
	fi

	if [[ ! -e  $data_dir/test.txt ]]; then
		mkdir -p $cache_dir/wiki103
		mkdir -p $data_dir
		echo "Downloading WikiText-103 via HuggingFace datasets..."
		python3 -c "
from datasets import load_dataset
import os
cache = '$cache_dir/wiki103'
ds = load_dataset('wikitext', 'wikitext-103-v1', trust_remote_code=True)
for split, fname in [('train','wiki.train.tokens'),('validation','wiki.valid.tokens'),('test','wiki.test.tokens')]:
    out = os.path.join(cache, fname)
    if not os.path.exists(out):
        with open(out, 'w', encoding='utf-8') as f:
            for row in ds[split]:
                line = row['text'].strip()
                if line:
                    f.write(line + '\n')
        print(f'Wrote {out}')
"
		python ./prepare_data.py -i $cache_dir/wiki103/wiki.train.tokens -o $data_dir/train.txt --max_seq_length $max_seq_length
		python ./prepare_data.py -i $cache_dir/wiki103/wiki.valid.tokens -o $data_dir/valid.txt --max_seq_length $max_seq_length
		python ./prepare_data.py -i $cache_dir/wiki103/wiki.test.tokens  -o $data_dir/test.txt  --max_seq_length $max_seq_length
	fi
}

setup_wiki_data

Task=RTD

init=$1
tag=$init
case ${init,,} in
	deberta-v3-xsmall-continue)
	# wget https://huggingface.co/microsoft/deberta-v3-xsmall/resolve/main/pytorch_model.generator.bin
	# wget https://huggingface.co/microsoft/deberta-v3-xsmall/resolve/main/pytorch_model.bin
	parameters=" --num_train_epochs 1 \
	--model_config rtd_xsmall.json \
	--warmup 10000 \
	--num_training_steps 100000 \
	--learning_rate 5e-5 \
	--train_batch_size 256 \
	--init_generator <TODO: generator checkpoint> \
	--init_discriminator <TODO: discriminator checkpoint> \
	--decoupled_training True \
	--fp16 True "
		;;
	deberta-v3-xsmall)
	parameters=" --num_train_epochs 1 \
	--model_config rtd_xsmall.json \
	--warmup 10000 \
	--learning_rate 3e-4 \
	--train_batch_size 64 \
	--decoupled_training True \
	--fp16 True "
		;;
	deberta-v3-small-continue)
	# wget https://huggingface.co/microsoft/deberta-v3-small/resolve/main/pytorch_model.generator.bin
	# wget https://huggingface.co/microsoft/deberta-v3-small/resolve/main/pytorch_model.bin
	parameters=" --num_train_epochs 1 \
	--model_config rtd_small.json \
	--warmup 10000 \
	--num_training_steps 100000 \
	--learning_rate 5e-5 \
	--train_batch_size 256 \
	--init_generator <TODO: generator checkpoint> \
	--init_discriminator <TODO: discriminator checkpoint> \
	--decoupled_training True \
	--fp16 True "
		;;
	deberta-v3-base)
	parameters=" --num_train_epochs 1 \
	--model_config rtd_base.json \
	--warmup 10000 \
	--learning_rate 1e-4 \
	--train_batch_size 256 \
	--decoupled_training True \
	--fp16 True "
		;;
	deberta-v3-large)
	parameters=" --num_train_epochs 1 \
	--model_config rtd_large.json \
	--warmup 10000 \
	--learning_rate 1e-4 \
	--train_batch_size 256 \
	--decoupled_training True \
	--fp16 True "
		;;
	recurrent-deberta-v3-large)
	# Recurrent DeBERTa-v3-Large: single shared layer (Layer 0) as discriminator.
	LAYER0_DISC=${LAYER0_DISC:-/tmp/layer0_disc.bin}
	if [[ ! -e $LAYER0_DISC ]]; then
		echo "Extracting Layer 0 weights from DeBERTa-v3-Large -> $LAYER0_DISC"
		python ../../experiments/language_model/extract_layer0.py --output $LAYER0_DISC
	fi
	parameters=" --num_train_epochs 1 \
	--model_config rtd_recurrent_large.json \
	--warmup 10000 \
	--learning_rate 1e-4 \
	--train_batch_size 64 \
	--accumulative_update 4 \
	--decoupled_training True \
	--fp16 True \
	--init_discriminator $LAYER0_DISC"
		;;
	*)
		echo "usage $0 <Pretrained model configuration>"
		echo "Supported configurations"
		echo "deberta-v3-xsmall - Pretrained DeBERTa v3 XSmall model with 9M backbone network parameters (12 layers, 256 hidden size) plus 32M embedding parameters(128k vocabulary size)"
		echo "deberta-v3-xsmall - Pretrained DeBERTa v3 Base model with 81M backbone network parameters (12 layers, 768 hidden size) plus 96M embedding parameters(128k vocabulary size)"
		echo "deberta-v3-xsmall - Pretrained DeBERTa v3 Large model with 288M backbone network parameters (24 layers, 1024 hidden size) plus 128M embedding parameters(128k vocabulary size)"
		exit 0
		;;
esac

python -m DeBERTa.apps.run --model_config config.json  \
	--tag $tag \
	--do_train \
	--num_training_steps ${NUM_STEPS:-1000000} \
	--max_seq_len $max_seq_length \
	--dump 10000 \
	--task_name $Task \
	--data_dir $data_dir \
	--vocab_path $cache_dir/spm.model \
	--vocab_type spm \
	--output_dir /tmp/ttonly/$tag/$task  $parameters
