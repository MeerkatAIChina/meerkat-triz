PATCH_NOTES.md — Meerkat-TRIZ v6 Qwen3.8-27B 换基座
=====================================================
创建时间: 2026-08-16 (DGX 本地时间)

结论: 无需补丁
-------------
Qwen3.8-27B (qwen3_5 混合线性注意力多模态架构, 64层, hidden 5120)
通过 transformers AutoModelForCausalLM 直接加载成功，未触发多模态/vision tower/MTP 层相关异常。

training 启动检查清单:
- 模型加载:  PASS (显存 50.1 GB, BF16)
- LoRA 挂载: PASS (trainable 466,911,232 / 27,362,909,696 = 1.7064%, r=64 alpha=128)
- 数据契约:  PASS (train=11096 / val=1050, 2048 截断率 0%)
- 断言 A (max_length=2048): PASS
- 断言 B (eval_steps==save_steps=100): PASS
- 断言 C (首 batch prompt 区段 labels 全 -100): PASS (loss 冒烟: prompt 46 token 全 -100)
- completion 区段无 ChatML 泄漏: PASS

无额外修补动作。

--- 更新记录 ---

[2026-08-16 15:25] 训练启动，正常推进。
[2026-08-16 18:34] 完成 step 500 eval，eval_loss=1.5616 (新低)，best checkpoint 保存。
[2026-08-16 ~18:35-20:25] 训练进程意外死亡（GPU 利用率降为 0%，ps 无进程）。
  - 死因排查: dmesg 无权限查看；日志无 OOM/异常；磁盘空间充足(13%)；内存充足(118GB可用)；系统未重启。
  - 推断: 可能为用户手动停止或信号终止。
[2026-08-16 20:29] 尝试从 checkpoint-500 续训，遇到 optimizer 数据类型错误:
  RuntimeError: expected dtype float for 'end' but got dtype c10::BFloat16
  根因: checkpoint-500 中 optimizer.pt 有 1984 个 BF16 张量（占2/3），Adam _multi_tensor_adam 要求 FP32。
  修补: 将 optimizer.pt 中所有 BF16 张量转换为 FP32，原始文件备份为 optimizer.pt.backup。
[2026-08-16 20:39] 使用修复后的 checkpoint-500 重新 resume，模型加载中。
