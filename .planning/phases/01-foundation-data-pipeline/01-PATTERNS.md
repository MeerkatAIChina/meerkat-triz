# Phase 01: Foundation & Data Pipeline - Pattern Map

**Mapped:** 2026-05-27
**Files analyzed:** 11
**Analogs found:** 10 / 11

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `config.py` (modify) | config | static | `ref/mongoose_ai_dgx/config.py` | exact |
| `requirements.txt` (modify) | config | static | `ref/mongoose_ai_dgx/requirements.txt` | exact |
| `README.md` (modify) | config | static | `ref/mongoose_ai_dgx/README.md` | exact |
| `utils/__init__.py` (modify) | config | static | `ref/mongoose_ai_dgx/utils/__init__.py` | exact |
| `utils/synthetic_pipeline.py` (new) | service | request-response | `ref/mongoose_ai_dgx/utils/data_utils.py` | role-match |
| `utils/pipeline_state.py` (new) | utility | file-I/O | `ref/mongoose_ai_dgx/utils/data_utils.py` | partial |
| `notebooks/02b_synthetic_generation.ipynb` (new) | component | batch | `ref/mongoose_ai_dgx/notebooks/02_data_preparation.ipynb` | role-match |
| `notebooks/01_download_and_setup.ipynb` (modify) | component | request-response | `ref/mongoose_ai_dgx/notebooks/01_download_and_setup.ipynb` | exact |
| `notebooks/02_data_preparation.ipynb` (modify) | component | request-response | `ref/mongoose_ai_dgx/notebooks/02_data_preparation.ipynb` | exact |
| `tests/test_synthetic_pipeline.py` (new) | test | request-response | None | no-analog |
| `tests/test_pipeline_state.py` (new) | test | file-I/O | None | no-analog |

## Pattern Assignments

### `config.py` (config, static) - MODIFY

**Analog:** `ref/mongoose_ai_dgx/config.py`

**Current lora_dropout pattern** (lines 61):
```python
"lora_dropout": 0.05,       # Dropout率
```

**Current target_modules pattern** (lines 56-60):
```python
"target_modules": [
    "q_proj", "k_proj", "v_proj", "o_proj",           # Gated Attention
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",  # Gated DeltaNet
    "gate_proj", "up_proj", "down_proj",              # MoE MLP
],
```

**Directory auto-creation pattern** (lines 18-20):
```python
for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, CHECKPOINTS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
```

**Config import pattern from notebooks** (from `01_download_and_setup.ipynb` cell-6):
```python
from config import (
    BASE_MODEL, MODELS_DIR, DATA_DIR, OUTPUTS_DIR,
    QLORA_CONFIG, DATA_CONFIG, BENCHMARK_CONFIG, HARDWARE_CONFIG
)
```

---

### `requirements.txt` (config, static) - MODIFY

**Analog:** `ref/mongoose_ai_dgx/requirements.txt`

**Current unpinned pattern** (lines 10-12):
```python
transformers>=4.45.0      # Qwen3.6需要4.45+
peft>=0.12.0              # 支持'all-linear'和rsLoRA
bitsandbytes>=0.43.0
```

**Current section structure** (lines 4-54):
- Core deep learning framework
- Large model fine-tuning
- Training and data processing
- Evaluation framework
- Data processing and scientific computing
- Visualization
- Utilities
- Jupyter environment
- Logging and monitoring
- Chinese segmentation

---

### `README.md` (config, static) - MODIFY

**Analog:** `ref/mongoose_ai_dgx/README.md`

**Current "all-linear" recommendation** (line 119):
```markdown
本套件已内置三种适配方案，**默认使用 `"all-linear"` 自动检测**：
```

**Code block pattern for target_modules** (lines 122-135):
```python
# 方案1: PEFT自动检测 (推荐, 默认)
target_modules = "all-linear"

# 方案2: 手动指定Qwen3.6模块列表
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",           # Gated Attention
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",  # GDN
    "gate_proj", "up_proj", "down_proj",              # MoE MLP
]
```

---

### `utils/__init__.py` (config, static) - MODIFY

**Analog:** `ref/mongoose_ai_dgx/utils/__init__.py`

**Import pattern** (lines 5-31):
```python
from .benchmark_utils import (
    run_lm_evaluation,
    run_triz_evaluation,
    run_performance_benchmark,
    aggregate_results,
)

from .data_utils import (
    load_raw_data,
    convert_to_chatml,
    create_synthetic_data,
    split_dataset,
    save_dataset,
    validate_chatml_format,
)

from .training_utils import (
    load_model_and_tokenizer,
    setup_qlora_config,
    prepare_qlora_model,
    setup_training_arguments,
    create_trainer,
    merge_and_save_model,
    save_adapter_only,
    find_all_linear_names,
    get_qwen36_target_modules,
)
```

**__all__ pattern** (lines 33-53):
```python
__all__ = [
    "run_lm_evaluation",
    "run_triz_evaluation",
    ...
]
```

---

### `utils/synthetic_pipeline.py` (service, request-response) - NEW

**Analog:** `ref/mongoose_ai_dgx/utils/data_utils.py` (for data loading/saving patterns)

**Imports pattern** (from `data_utils.py` lines 1-14):
```python
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datasets import Dataset, DatasetDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

**File I/O pattern** (from `data_utils.py` lines 45-62):
```python
def load_raw_data(data_dir: str) -> Dict[str, List[Dict]]:
    data_dir = Path(data_dir)
    subsets = {}
    
    if not data_dir.exists():
        logger.warning(f"数据目录不存在: {data_dir}，将创建示例数据")
        return create_sample_data()
    
    for json_file in sorted(data_dir.glob("*.json")):
        subset_name = json_file.stem
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            subsets[subset_name] = data
            logger.info(f"加载 {subset_name}: {len(data)} 条样本")
        except Exception as e:
            logger.error(f"加载 {json_file} 失败: {e}")
    
    return subsets
```

**JSON save pattern** (from `data_utils.py` lines 365-373):
```python
def save_dataset(dataset: DatasetDict, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_dataset in dataset.items():
        split_path = output_path / f"{split_name}.jsonl"
        split_dataset.to_json(str(split_path))
        logger.info(f"保存 {split_name}: {len(split_dataset)} 条 → {split_path}")
```

**API client pattern** (from RESEARCH.md):
```python
from openai import OpenAI
import time

class MoonshotSyntheticClient:
    def __init__(self, api_key: str, model: str = "moonshot-v1-8k"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1"
        )
        self.model = model
        self.rpm = 3  # Tier 0 default
        self.min_interval = 60.0 / self.rpm
        self.last_request_time = 0
```

---

### `utils/pipeline_state.py` (utility, file-I/O) - NEW

**Analog:** `ref/mongoose_ai_dgx/utils/data_utils.py` (for JSON file handling)

**JSON load/save pattern** (from `data_utils.py` lines 99-111):
```python
json_path = os.path.join(data_dir, "sample_data.json")

if os.path.exists(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            sample_data = json.load(f)
        total = sum(len(v) for v in sample_data.values())
        logger.info(f"从JSON加载示例数据: {total} 条，覆盖 {len(sample_data)} 个子集")
        return sample_data
    except Exception as e:
        logger.warning(f"加载JSON数据失败 ({e})，使用回退数据")
```

**Path creation pattern** (from `data_utils.py` lines 367-368):
```python
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)
```

**Registry pattern** (from RESEARCH.md):
```python
import json
from pathlib import Path
from datetime import datetime

class PipelineState:
    def __init__(self, state_file: str = "data/processed/pipeline_state.json"):
        self.state_file = Path(state_file)
        self.state = self._load()

    def _load(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {"artifacts": [], "version": "1.0"}

    def register(self, name: str, path: str, artifact_type: str, metadata: dict = None):
        artifact = {
            "name": name,
            "path": str(path),
            "type": artifact_type,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.state["artifacts"] = [a for a in self.state["artifacts"] if a["name"] != name]
        self.state["artifacts"].append(artifact)
        self._save()
```

---

### `notebooks/02b_synthetic_generation.ipynb` (component, batch) - NEW

**Analog:** `ref/mongoose_ai_dgx/notebooks/02_data_preparation.ipynb`

**Notebook cell structure pattern** (from `02_data_preparation.ipynb`):
- Cell 0: Markdown title/header
- Cell 1: Markdown section header
- Cell 2: Python imports + sys.path.append + function calls
- Cell 3: Markdown section header
- Cell 4: Python data inspection
- Cell 5: Markdown section header
- Cell 6: Python processing with config imports
- Cell 7: Markdown section header
- Cell 8: Python display sample
- Cell 9: Markdown section header
- Cell 10: Python validation
- Cell 11: Markdown section header
- Cell 12: Python save output
- Cell 13: Markdown next steps

**sys.path pattern** (from `02_data_preparation.ipynb` cell-2):
```python
import sys
sys.path.append('/home/meerkat/mongoose_ai')

from utils.data_utils import load_raw_data, create_sample_data
from config import DATA_DIR, DATA_CONFIG
```

**Config import pattern** (from `02_data_preparation.ipynb` cell-6):
```python
from config import DATA_CONFIG, BASE_MODEL, MODELS_DIR
```

**Model loading for processing pattern** (from `02_data_preparation.ipynb` cell-6):
```python
from utils.training_utils import load_model_and_tokenizer

model_path = os.path.join(MODELS_DIR, BASE_MODEL.split('/')[-1])
_, tokenizer = load_model_and_tokenizer(
    model_name_or_path=model_path,
    quantization_config=None,
    device_map='cpu',
)
```

---

### `notebooks/01_download_and_setup.ipynb` (component, request-response) - MODIFY

**Analog:** `ref/mongoose_ai_dgx/notebooks/01_download_and_setup.ipynb`

**Hardware check pattern** (cell-2):
```python
import torch
import os

print("="*60)
print("DGX Spark 硬件检查")
print("="*60)
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  总内存: {props.total_memory / 1024**3:.1f} GB")
```

**Dependency install pattern** (cell-4):
```python
!pip install -q transformers peft bitsandbytes accelerate datasets trl
!pip install -q evaluate scikit-learn matplotlib seaborn tqdm
!pip install -q jieba huggingface-hub
```

**Config load pattern** (cell-6):
```python
import sys
sys.path.append('/home/meerkat/mongoose_ai')

from config import (
    BASE_MODEL, MODELS_DIR, DATA_DIR, OUTPUTS_DIR,
    QLORA_CONFIG, DATA_CONFIG, BENCHMARK_CONFIG, HARDWARE_CONFIG
)
```

---

### `notebooks/02_data_preparation.ipynb` (component, request-response) - MODIFY

**Analog:** `ref/mongoose_ai_dgx/notebooks/02_data_preparation.ipynb`

**Data loading pattern** (cell-2):
```python
import sys
sys.path.append('/home/meerkat/mongoose_ai')

from utils.data_utils import load_raw_data, create_sample_data
from config import DATA_DIR, DATA_CONFIG

raw_data = load_raw_data(DATA_CONFIG['raw_data_dir'])
```

**ChatML conversion pattern** (cell-6):
```python
from utils.data_utils import convert_to_chatml
from utils.training_utils import load_model_and_tokenizer
from config import DATA_CONFIG, BASE_MODEL, MODELS_DIR
import os

model_path = os.path.join(MODELS_DIR, BASE_MODEL.split('/')[-1])
_, tokenizer = load_model_and_tokenizer(
    model_name_or_path=model_path,
    quantization_config=None,
    device_map='cpu',
)

dataset = convert_to_chatml(
    data=raw_data,
    tokenizer=tokenizer,
    system_message=DATA_CONFIG['chatml']['system_message']
)
```

**Validation pattern** (cell-10):
```python
from utils.data_utils import validate_chatml_format

report = validate_chatml_format(dataset)

print(f"总样本数: {report['total_samples']}")
print(f"有效样本: {report['valid_samples']}")
print(f"无效样本: {report['invalid_samples']}")
```

**Save pattern** (cell-12):
```python
from utils.data_utils import save_dataset

processed_dir = DATA_CONFIG['processed_data_dir']
save_dataset(dataset, processed_dir)
```

---

## Shared Patterns

### Logging
**Source:** `ref/mongoose_ai_dgx/utils/data_utils.py` (lines 13-14)
**Apply to:** All new/modified Python files
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

### Path Handling
**Source:** `ref/mongoose_ai_dgx/config.py` (lines 9-20)
**Apply to:** All new files that reference paths
```python
from pathlib import Path

BASE_DIR = Path("/home/meerkat/mongoose_ai")
DATA_DIR = BASE_DIR / "data"
# Auto-create directories
for d in [DATA_DIR, ...]:
    d.mkdir(parents=True, exist_ok=True)
```

### Config Import
**Source:** `ref/mongoose_ai_dgx/notebooks/01_download_and_setup.ipynb` (cell-6)
**Apply to:** All notebooks
```python
import sys
sys.path.append('/home/meerkat/mongoose_ai')

from config import (
    BASE_MODEL, MODELS_DIR, DATA_DIR, OUTPUTS_DIR,
    QLORA_CONFIG, DATA_CONFIG, BENCHMARK_CONFIG, HARDWARE_CONFIG
)
```

### Error Handling
**Source:** `ref/mongoose_ai_dgx/utils/data_utils.py` (lines 54-61)
**Apply to:** All service/utility files
```python
try:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    subsets[subset_name] = data
    logger.info(f"加载 {subset_name}: {len(data)} 条样本")
except Exception as e:
    logger.error(f"加载 {json_file} 失败: {e}")
```

### JSON Serialization
**Source:** `ref/mongoose_ai_dgx/utils/benchmark_utils.py` (lines 74-76)
**Apply to:** All files that save JSON results
```python
import json
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_file = output_path / f"results_{timestamp}.json"
with open(result_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

### Type Hints
**Source:** `ref/mongoose_ai_dgx/utils/data_utils.py` (lines 19, 65, 144)
**Apply to:** All new Python modules
```python
from typing import Dict, List, Optional, Tuple, Any

def load_raw_data(data_dir: str) -> Dict[str, List[Dict]]:
def create_sample_data(data_dir: Optional[str] = None) -> Dict[str, List[Dict]]:
def convert_to_chatml(
    data: Dict[str, List[Dict]],
    tokenizer=None,
    system_message: Optional[str] = None
) -> DatasetDict:
```

## No Analog Found

Files with no close match in the codebase (planner should use RESEARCH.md patterns instead):

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `tests/test_synthetic_pipeline.py` | test | request-response | No test files exist in the codebase yet |
| `tests/test_pipeline_state.py` | test | file-I/O | No test files exist in the codebase yet |

## Metadata

**Analog search scope:** `ref/mongoose_ai_dgx/` (config.py, requirements.txt, utils/*.py, notebooks/*.ipynb, README.md, data/sample_data.json)
**Files scanned:** 10
**Pattern extraction date:** 2026-05-27
