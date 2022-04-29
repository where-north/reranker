

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_rerank_encoder.py \
--do_train \
--epoch 3 \
--use_multi_gpu \
--lr 4e-5 \
--model_type nezha_base \
--batch_size 32 \
#--do_adversarial \

#python train_rerank_encoder.py \
#--do_train \
#--epoch 3 \
#--gpu_id 3


#for n in $(seq 0 3);
# do
#     if [ $n -eq 0 ]
#         then
# 	    python train_rerank_encoder.py \
#              --do_predict --batch_size 256 \
#              --test_data_path ./data/data4rerank_0 \
#              --gpu_id 0 \
#              --model_timestamp 2022-04-27_15_51_29 &
#     elif [ $n -eq 1 ]
#         then
# 	    python train_rerank_encoder.py \
#              --do_predict --batch_size 256 \
#              --test_data_path ./data/data4rerank_1 \
#              --gpu_id 1 \
#              --model_timestamp 2022-04-27_15_51_29 &
#     elif [ $n -eq 2 ]
# 	then
# 	    python train_rerank_encoder.py \
#              --do_predict --batch_size 256 \
#              --test_data_path ./data/data4rerank_2 \
#              --gpu_id 2 \
#              --model_timestamp 2022-04-27_15_51_29 &
#     elif [ $n -eq 3 ]
# 	then
#       python train_rerank_encoder.py \
#              --do_predict --batch_size 256 \
#              --test_data_path ./data/data4rerank_3 \
#              --gpu_id 3 \
#              --model_timestamp 2022-04-27_15_51_29 &
#     fi
# done
# wait

#for n in $(seq 0 1);
# do
#     if [ $n -eq 0 ]
#         then
# 	    python train_rerank_encoder.py \
#              --do_predict --batch_size 256 \
#              --test_data_path ./data/data4rerank_0 \
#              --gpu_id 0 \
#              --model_timestamp 2022-04-27_15_51_29 &
#     elif [ $n -eq 1 ]
#         then
# 	    python train_rerank_encoder.py \
#              --do_predict --batch_size 256 \
#              --test_data_path ./data/data4rerank_1 \
#              --gpu_id 1 \
#              --model_timestamp 2022-04-27_15_51_29 &
#     fi
# done
# wait
#
# for n in $(seq 0 1);
# do
#     if [ $n -eq 0 ]
#         then
# 	    python train_rerank_encoder.py \
#              --do_predict --batch_size 256 \
#              --test_data_path ./data/data4rerank_2 \
#              --gpu_id 0 \
#              --model_timestamp 2022-04-27_15_51_29 &
#     elif [ $n -eq 1 ]
#         then
# 	    python train_rerank_encoder.py \
#              --do_predict --batch_size 256 \
#              --test_data_path ./data/data4rerank_3 \
#              --gpu_id 1 \
#              --model_timestamp 2022-04-27_15_51_29 &
#     fi
# done
# wait


# 重排序验证集
#python train_rerank_encoder.py \
#        --do_predict --batch_size 256 \
#        --test_data_path ./data/val_recall_data4rerank.json \
#        --gpu_id 0 \
#        --model_timestamp 2022-04-28_07_23_35
