# python eval_debug_data.py \
# --type local \
# --ckpt /data/haoxiang/logs/airexo_task0012_mask_aware/policy_last.ckpt \
# --calib_rise2 /home/haoxiang/rise2_calib_1770288769108.npy \
# --config configs/dual_teleop_dino.yaml \
# --rgb /data/haoxiang/260316_maskaware_task0012_visdebug/rgb/step_000121.png \
# --depth /data/haoxiang/260316_maskaware_task0012_visdebug/depth/step_000121.png \
# --mask /data/haoxiang/260316_maskaware_task0012_visdebug/mask/step_000121.png

python eval_debug_data.py \
--type local \
--ckpt /data/haoxiang/logs/airexo_task0012_mask_aware/policy_last.ckpt \
--calib_rise2 /data/haoxiang/data/airexo2/task_0012/calib_new/1736232303704.npy \
--config configs/dual_teleop_dino.yaml \
--rgb /data/haoxiang/data/airexo2/task_0012/train/scene_0001/cam_105422061350/color/1736245301149.png \
--depth /data/haoxiang/data/airexo2/task_0012/train/scene_0001/cam_105422061350/depth/1736245301149.png \
--mask /data/haoxiang/data/airexo2/task_0012/merged_masks/scene_0001/00180.png

