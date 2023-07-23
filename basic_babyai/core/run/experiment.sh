python3 ../scripts/train_rl.py --env BabyAI-OpenDoorsOrderN4-v0 \
--test_env BabyAI-OpenDoorsOrderN4-v0 \
--seed=1 \
--frames=300000000 \
--instr_to_rule_mode='linear' --instr_arch='gru' \
--instr_dim=64 \
--memory_dim 1024 \
--model "FiLM" --arch='expert_filmcnn' \
--procs=16 --batch_size=1280 --frames_per_proc=80 --recurrence=20 \
--save_interval=500 \
--use_compositional_split 

