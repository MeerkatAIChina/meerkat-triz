import json
p = "/home/chinux/jupyterlab/meerkatai/checkpoints/qlora_triz_v6_qwen38/checkpoint-500/trainer_state.json"
d = json.load(open(p))
print(f"global_step: {d.get('global_step')}")
print(f"epoch: {d.get('epoch')}")
print(f"best_model_checkpoint: {d.get('best_model_checkpoint')}")
print(f"is_local_process_zero: {d.get('is_local_process_zero')}")
print(f"log_history entries: {len(d.get('log_history', []))}")

# 查看最后几条 log
for entry in d.get('log_history', [])[-5:]:
    print(f"  {entry}")
