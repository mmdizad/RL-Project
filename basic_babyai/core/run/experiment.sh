python3 ../scripts/train_rl.py --env BabyAI-GoToSeqS5R2-v0 \
--test_env BabyAI-GoToSeqS5R2-v0 \
--seed=42 \
--frames=20000000 \
--instr_to_rule_mode='linear' --instr_arch='gru' \
--instr_dim=64 \
--memory_dim 1024 \
--model "gotoseq_min" --arch='cnn1' \
--procs=16 --batch_size=1280 --frames_per_proc=80 --recurrence=20 \
--save_interval=500 \
--use_compositional_split \
--apply_aux \
--x_clip_coef 0.01 \
--x_clip_temp 1 \
--apply_instruction_tracking \
--instruction_tracking_threshold 2 \