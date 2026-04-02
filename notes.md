pixi run python scripts/splatam.py configs/replica/replica_room0.py 2>&1 | tee experiments/Replica/room0_0.log
pixi run python viz_scripts/render_frames.py configs/replica/replica_office2.py --eval_every 1 --save_frames
pixi run python scripts/vlm_evaluation.py --scene room0 --eval_dir experiments/Replica/room0_0/eval --num_views 40 --eval_every 1 --model gpt-4o-mini
pixi run python scripts/vlm_evaluation_single_frame.py --scene room0 --eval_dir experiments/Replica/room0_0/eval --frame_idx 138 --model gpt-4o-mini
pixi run python scripts/bev_from_gaussians.py   --params_path experiments/Replica/room0_0/params.npz   --output_dir experiments/Replica/room0_0/bev   --resolution 512
