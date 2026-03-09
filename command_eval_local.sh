PYTHONPATH=/home/haoxiang/rise2_mask_aware/airexo:${PYTHONPATH} \
python eval_rise2_dev_dataset.py \
--type local \
--ckpt /data/haoxiang/logs/airexo_mask_task0013/policy_last.ckpt \
--calib_airexo /data/haoxiang/data/airexo2/task_0013/calib/1737548651048.npy \
--calib_rise2 /data/haoxiang/data/airexo2/task_0013/calib/rise2_calib_1737548651048.npy \
--config configs/dual_teleop_dino.yaml

