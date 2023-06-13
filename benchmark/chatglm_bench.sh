n_layers=1
precision="fp16"
batch_size=1
seqlens=(8 16 32 64 128 256 512 1024)
head_num=32
head_size=128
avg_seqlen=0
gpu_card=1
export CUDA_VISIBLE_DEVICES=${gpu_card}

for seqlen in ${seqlens[@]}; do
  echo "seq_len: ${seqlen}"
  python /home/nfs_data/zhanggh/ByteEngine/unit_test/python_scripts/bert_transformer_test.py ${batch_size} ${seqlen} ${head_num} ${head_size} --avg_seqlen=${avg_seqlen} --n_layers=${n_layers} --dtype=${precision} --lib_path=/home/nfs_data/zhanggh/ByteEngine/build/lib/libths_bytetransformer.so --iters=1
done