python3 ../scripts/train_rl.py --env BabyAI-GoToSeqS5R2-v0 \
--test_env BabyAI-GoToSeqS5R2-v0 \
--seed=1 \
--frames=300000000 \
--instr_to_rule_mode='linear' --instr_arch='gru' \
--instr_dim=64 \
--memory_dim 1024 \
--x_clip_coef 0.1 \
--x_clip_temp 1.0 \
--model "FiLM" --arch='expert_filmcnn' \
--procs=16 --batch_size=1280 --frames_per_proc=80 --recurrence=20 \
--save_interval=500 \
--use_compositional_split 

