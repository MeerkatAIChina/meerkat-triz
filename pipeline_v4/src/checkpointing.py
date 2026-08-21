"""
pipeline_v4 checkpoint 管理: best-ckpt 精确归因、防轮转另存、严格验证、安全发货。

修复的旧管线缺陷:
  1. best checkpoint 丢失: save_total_limit=3 把真正最优 checkpoint 轮转删除。
     → BestCheckpointCallback 在 eval 创新低时立即把对应 checkpoint 复制到
       <output_dir>/best/, 不参与 Trainer 轮转。
  2. 归因错误: 旧 find_best_checkpoint 扫描幸存 checkpoint 的 trainer_state.json
     全量 log_history, 把别的 step 的 eval_loss 记到自己头上。
     → 新实现只采纳 log_history 中 step == 该 checkpoint 步数 的 eval_loss 条目。
  3. 验证形同虚设: 旧验证只查文件存在 + ≥1MB + 前向能跑, 全零 lora_B 也 PASSED。
     → 新验证断言: 全部 lora_B 张量非零、dtype 为 BF16、sha256 记录、
       adapter_config 完整。
  4. 发货 fallback: 任何 best checkpoint 不可用的情况, 一律复制 Trainer 落盘的
     checkpoint 目录文件 (Trainer 落盘是好的), 绝不保存训练内存态。

本模块刻意保持 torch-free (纯标准库 + 手解 safetensors header),
保证 --help / dry-run 秒回, 也保证验证逻辑可在任何环境独立运行。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

# Trainer 落盘的 LoRA checkpoint 中构成"可发货适配器"的文件
ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors"]
# tokenizer 相关文件: best checkpoint 里通常有 (Trainer 保存 processing_class),
# 缺失时从末步 checkpoint 补齐
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
]

MIN_ADAPTER_BYTES = 1 * 1024 * 1024  # 旧验证的 ≥1MB 下限保留为必要条件 (非充分)


def log_marked(msg: str, log_fn: Callable[[str], None] = print) -> None:
    """显眼标记日志, 用于 best 更新 / 验证结果 / 发货等关键事件。"""
    log_fn(f"★★★ {msg} ★★★")


# ---------------------------------------------------------------------------
# sha256
# ---------------------------------------------------------------------------

def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_manifest(directory: str, filenames: Optional[List[str]] = None) -> Dict[str, str]:
    """记录目录内指定文件 (默认全部常规文件, 不含子目录) 的 sha256。"""
    manifest = {}
    names = filenames if filenames is not None else sorted(os.listdir(directory))
    for name in names:
        p = os.path.join(directory, name)
        if os.path.isfile(p):
            manifest[name] = sha256_file(p)
    return manifest


# ---------------------------------------------------------------------------
# safetensors 手解 (torch-free): header 为文件起始 8 字节 little-endian
# uint64 头长 + JSON 头, 张量数据紧随其后按 data_offsets 寻址
# ---------------------------------------------------------------------------

def _read_safetensors_header(path: str) -> Tuple[Dict[str, Any], int]:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: 文件过小, 不是合法 safetensors")
        (header_len,) = struct.unpack("<Q", raw)
        header = json.loads(f.read(header_len).decode("utf-8"))
    return header, 8 + header_len


def _tensor_bytes_nonzero(path: str, data_start: int, offsets: List[int],
                          chunk_size: int = 1 << 20) -> bool:
    """逐块读张量原始字节, 任一非零字节即 True。

    BF16 的全零张量每个元素为 0x0000 → 原始字节全零;
    反之任一非零值 (含 -0.0=0x8000) 都有非零字节。
    """
    begin, end = offsets
    with open(path, "rb") as f:
        f.seek(data_start + begin)
        remaining = end - begin
        while remaining > 0:
            chunk = f.read(min(chunk_size, remaining))
            if not chunk:
                break
            if any(chunk):
                return True
            remaining -= len(chunk)
    return False


# ---------------------------------------------------------------------------
# 验证 (修复点 3)
# ---------------------------------------------------------------------------

def validate_adapter_dir(adapter_dir: str) -> Dict[str, Any]:
    """
    完整验证一个待发货适配器目录。返回:
        {
          "status": "PASSED" | "FAILED",
          "checks": [ {"name": ..., "status": ..., "detail": ...}, ... ],
          "sha256": {filename: hexdigest},
          "tensor_stats": {...},
        }
    任何一项 FAILED → 总体 FAILED。
    """
    checks: List[Dict[str, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        checks.append({
            "name": name,
            "status": "PASSED" if ok else "FAILED",
            "detail": detail,
        })
        return ok

    # ---- 1. 必需文件存在 ----
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    files_ok = True
    for p in (config_path, weights_path):
        files_ok &= check(f"exists:{os.path.basename(p)}", os.path.isfile(p),
                          p if os.path.isfile(p) else "缺失")

    # ---- 2. adapter_config 完整 ----
    config = None
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
            missing_keys = [k for k in ("r", "lora_alpha", "target_modules", "peft_type")
                            if k not in config]
            check("adapter_config:完整",
                  not missing_keys and isinstance(config.get("target_modules"), list),
                  f"missing={missing_keys}" if missing_keys else
                  f"r={config.get('r')} alpha={config.get('lora_alpha')} "
                  f"targets={len(config.get('target_modules', []))}")
        except Exception as e:
            check("adapter_config:可解析", False, f"{type(e).__name__}: {e}")

    # ---- 3. 权重文件大小下限 (必要条件) ----
    if os.path.isfile(weights_path):
        size = os.path.getsize(weights_path)
        check("weights:大小>=1MB", size >= MIN_ADAPTER_BYTES, f"{size / 1e6:.1f} MB")

    # ---- 4. safetensors header / dtype / 非零 ----
    tensor_stats: Dict[str, Any] = {}
    if os.path.isfile(weights_path):
        try:
            header, data_start = _read_safetensors_header(weights_path)
            tensors = {k: v for k, v in header.items() if k != "__metadata__"}
            lora_a = {k: v for k, v in tensors.items() if "lora_A" in k}
            lora_b = {k: v for k, v in tensors.items() if "lora_B" in k}
            tensor_stats = {
                "total_tensors": len(tensors),
                "lora_A_tensors": len(lora_a),
                "lora_B_tensors": len(lora_b),
                "dtypes": sorted({v.get("dtype", "?") for v in tensors.values()}),
            }
            check("weights:张量非空", len(tensors) > 0, f"{len(tensors)} tensors")
            check("weights:lora_B存在", len(lora_b) > 0, f"{len(lora_b)} lora_B tensors")
            if lora_a and lora_b:
                check("weights:lora_A/B数量一致", len(lora_a) == len(lora_b),
                      f"A={len(lora_a)} B={len(lora_b)}")

            # 全部 lora_B dtype 必须为 BF16 (事故中 fallback 内存态落盘为 F32)
            bad_dtype = [k for k, v in lora_b.items() if v.get("dtype") != "BF16"]
            check("lora_B:dtype全BF16", not bad_dtype and len(lora_b) > 0,
                  "全部 BF16" if not bad_dtype else
                  f"{len(bad_dtype)} 个非BF16, 例: {bad_dtype[:3]}")

            # 全部 lora_B 必须非零 (事故核心: 310 个 lora_B 全零)
            zero_tensors = []
            for name, meta in lora_b.items():
                if not _tensor_bytes_nonzero(weights_path, data_start, meta["data_offsets"]):
                    zero_tensors.append(name)
            check("lora_B:全部非零", not zero_tensors and len(lora_b) > 0,
                  f"{len(lora_b)}/{len(lora_b)} 非零" if not zero_tensors else
                  f"{len(zero_tensors)}/{len(lora_b)} 全零, 例: {zero_tensors[:3]}")
        except Exception as e:
            check("weights:header可解析", False, f"{type(e).__name__}: {e}")

    # ---- 5. sha256 记录 (信息性, 恒 PASSED) ----
    manifest = sha256_manifest(adapter_dir) if os.path.isdir(adapter_dir) else {}
    check("sha256:已记录", bool(manifest), f"{len(manifest)} 个文件")

    status = "PASSED" if all(c["status"] == "PASSED" for c in checks) else "FAILED"
    return {
        "status": status,
        "checks": checks,
        "sha256": manifest,
        "tensor_stats": tensor_stats,
    }


# ---------------------------------------------------------------------------
# best checkpoint 发现 (修复点 2: 按 eval step 精确归因)
# ---------------------------------------------------------------------------

def _checkpoint_step(name: str) -> Optional[int]:
    m = re.fullmatch(r"checkpoint-(\d+)", name)
    return int(m.group(1)) if m else None


def eval_loss_at_step(state: Dict[str, Any], step: int) -> Optional[float]:
    """只采纳 log_history 中 step 精确匹配的 eval_loss (防止把别人的 eval 记到自己头上)。"""
    loss = None
    for entry in state.get("log_history", []):
        if "eval_loss" in entry and entry.get("step") == step:
            loss = entry["eval_loss"]
    return loss


def find_best_checkpoint(ckpt_dir: str) -> Optional[Dict[str, Any]]:
    """
    在 ckpt_dir 下按"该 checkpoint 自己 step 的 eval_loss"精确归因找最优。
    返回 {"path", "step", "eval_loss"} 或 None (无任何可归因 eval 记录)。
    """
    candidates = []
    if not os.path.isdir(ckpt_dir):
        return None
    for name in sorted(os.listdir(ckpt_dir)):
        step = _checkpoint_step(name)
        if step is None:
            continue
        state_path = os.path.join(ckpt_dir, name, "trainer_state.json")
        if not os.path.isfile(state_path):
            continue
        try:
            with open(state_path) as f:
                state = json.load(f)
        except Exception:
            continue
        loss = eval_loss_at_step(state, step)
        if loss is not None:
            candidates.append({"path": os.path.join(ckpt_dir, name),
                               "step": step, "eval_loss": loss})
    if not candidates:
        return None
    return min(candidates, key=lambda c: c["eval_loss"])


def last_checkpoint(ckpt_dir: str) -> Optional[Dict[str, Any]]:
    """末步 checkpoint (兜底 fallback)。"""
    if not os.path.isdir(ckpt_dir):
        return None
    steps = [s for s in (_checkpoint_step(n) for n in os.listdir(ckpt_dir)) if s is not None]
    if not steps:
        return None
    step = max(steps)
    return {"path": os.path.join(ckpt_dir, f"checkpoint-{step}"), "step": step,
            "eval_loss": None}


# ---------------------------------------------------------------------------
# BestCheckpointCallback 工厂 (修复点 1)
# ---------------------------------------------------------------------------

def build_best_checkpoint_callback(trainer_callback_base, output_dir: str,
                                   log_fn: Callable[[str], None] = print):
    """
    构造 BestCheckpointCallback 类 (基类 transformers.TrainerCallback 由调用方传入,
    以保持本模块 torch-free / --help 秒回)。

    行为:
      - on_evaluate: eval_loss 创新低 → 若 checkpoint-{step} 已落盘立即复制到
        <output_dir>/best/; 否则记 pending, 待 on_save 补存 (eval/save 不对齐兜底;
        v4 config 强制 eval_steps == save_steps, 正常路径下 eval 后同 step 即 save)。
      - on_save: 有 pending 且本 step checkpoint 已落盘 → 复制为 best-candidate。
      - best/ 不参与 Trainer 的 save_total_limit 轮转, 永不丢失。
    """

    class BestCheckpointCallback(trainer_callback_base):
        def __init__(self):
            self.output_dir = output_dir
            self.best_dir = os.path.join(output_dir, "best")
            self.best_eval_loss: Optional[float] = None
            self.best_eval_step: Optional[int] = None
            self.pending_step: Optional[int] = None
            # 逐条促销记录, 写入 adapter_info.json 保证可审计
            self.promotion_history: List[Dict[str, Any]] = []

        def _promote(self, src_dir: str, eval_step: int, eval_loss: float,
                     trigger: str) -> None:
            os.makedirs(self.best_dir, exist_ok=True)
            shutil.copytree(src_dir, self.best_dir, dirs_exist_ok=True)
            record = {
                "eval_step": eval_step,
                "eval_loss": eval_loss,
                "promoted_from": src_dir,
                "trigger": trigger,
                "time": datetime.now().isoformat(timespec="seconds"),
            }
            self.promotion_history.append(record)
            log_marked(
                f"NEW BEST: eval_loss={eval_loss:.6f} @ eval_step={eval_step} "
                f"→ 已另存 {self.best_dir} (trigger={trigger})",
                log_fn,
            )

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            metrics = metrics or {}
            loss = metrics.get("eval_loss")
            if loss is None:
                return
            step = state.global_step
            if self.best_eval_loss is None or loss < self.best_eval_loss:
                self.best_eval_loss = loss
                self.best_eval_step = step
                ckpt = os.path.join(self.output_dir, f"checkpoint-{step}")
                if os.path.isdir(ckpt):
                    self._promote(ckpt, step, loss, trigger="on_evaluate")
                    self.pending_step = None
                else:
                    # eval step 没有落盘 checkpoint (eval/save 不对齐), 下一个 save 点补存
                    self.pending_step = step
                    log_fn(
                        f"[best-ckpt] eval_loss 新低 {loss:.6f} @ step={step}, "
                        f"但该步无落盘 checkpoint, 将在下一个 save 点补存为 best-candidate"
                    )

        def on_save(self, args, state, control, **kwargs):
            step = state.global_step
            if self.pending_step is None:
                return
            ckpt = os.path.join(self.output_dir, f"checkpoint-{step}")
            if os.path.isdir(ckpt):
                self._promote(ckpt, self.pending_step, self.best_eval_loss,
                              trigger=f"on_save(deferred, saved_step={step})")
                self.pending_step = None

        def on_train_end(self, args, state, control, **kwargs):
            if self.best_eval_step is None:
                log_fn("[best-ckpt] 训练中未记录任何 eval_loss, best/ 为空")
            else:
                log_marked(
                    f"训练结束: best eval_loss={self.best_eval_loss:.6f} "
                    f"@ eval_step={self.best_eval_step}, best/ 共促销 "
                    f"{len(self.promotion_history)} 次",
                    log_fn,
                )

    return BestCheckpointCallback


# ---------------------------------------------------------------------------
# 安全发货 (修复点 4): 只复制文件, 绝不保存内存态
# ---------------------------------------------------------------------------

def ship_adapter(best_dir: str, dest_dir: str,
                 supplement_dirs: Optional[List[str]] = None,
                 log_fn: Callable[[str], None] = print) -> List[Dict[str, str]]:
    """
    把 best checkpoint 的适配器文件复制到 dest_dir (models/meerkat_triz_adapter_v4)。
    tokenizer 等必要文件若 best 中缺失, 依次从 supplement_dirs (如末步 checkpoint) 补齐。
    返回逐文件复制记录 [{file, from}], 供 adapter_info.json 审计。
    """
    os.makedirs(dest_dir, exist_ok=True)
    copied: List[Dict[str, str]] = []
    for name in ADAPTER_FILES + TOKENIZER_FILES:
        src = os.path.join(best_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dest_dir, name))
            copied.append({"file": name, "from": best_dir})
    for name in TOKENIZER_FILES:
        dst = os.path.join(dest_dir, name)
        if os.path.isfile(dst):
            continue
        for d in supplement_dirs or []:
            src = os.path.join(d, name)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                copied.append({"file": name, "from": d})
                log_fn(f"[ship] {name} 在 best 中缺失, 已从 {d} 补齐")
                break
    log_marked(f"发货: {len(copied)} 个文件 → {dest_dir} (来源全部为磁盘 checkpoint 文件)", log_fn)
    return copied
