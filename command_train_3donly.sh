# # teleop
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_addr 127.0.0.1 --master_port 23333 --nproc_per_node 4 --nnodes 1 --node_rank 0 train.py --data_path data/collect_toys --ckpt_dir logs/collect_toys --config configs/dual_teleop_dino.yaml

# # wild
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_addr 127.0.0.1 --master_port 23333 --nproc_per_node 4 --nnodes 1 --node_rank 0 train.py --data_path data/collect_toys_wild --ckpt_dir logs/collect_toys_wild --config configs/dual_wild_dino.yaml

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_addr 127.0.0.1 --master_port 23333 --nproc_per_node 4 --nnodes 1 --node_rank 0 train.py --data_path /data/haoxiang/data/airexo2/task_0013 --ckpt_dir /data/haoxiang/logs/airexo_mask_task0013 --config configs/dual_teleop_dino.yaml

# CUDA_VISIBLE_DEVICES=0 torchrun --master_addr 127.0.0.1 --master_port 23333 --nproc_per_node 1 --nnodes 1 --node_rank 0 train.py --data_path /data/haoxiang/data/airexo2/task_0013 --ckpt_dir /data/haoxiang/logs/airexo_mask_task0013 --config configs/dual_teleop_dino.yaml

# export LIBGL_ALWAYS_SOFTWARE=1
# export MPLBACKEND=Agg # 固定无头后端：MPLBACKEND=Agg。
# # 远程服务器无窗口时不依赖 DISPLAY，vis 会离线保存到 config.data.vis_save_dir
# CUDA_VISIBLE_DEVICES=0 torchrun --master_addr 127.0.0.1 --master_port 23333 --nproc_per_node 1 --nnodes 1 --node_rank 0 train.py --data_path /data/haoxiang/data/airexo2/task_0013 --ckpt_dir /data/haoxiang/logs/airexo_mask_task0013 --config configs/dual_teleop_dino.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_addr 127.0.0.1 --master_port 23333 --nproc_per_node 4 --nnodes 1 --node_rank 0 train_3donly.py --data_path /data/haoxiang/data/airexo2/task_0013 --ckpt_dir /data/haoxiang/logs/airexo_mask_task0013_3donly --config configs/dual_teleop_dino_3donly.yaml

