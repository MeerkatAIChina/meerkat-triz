# MANIFEST — 训练数据集清单（模板）

> 每版训练集必须在训练前填写本文件的一份副本（建议命名为 `MANIFEST_<数据集版本>.md`，如 `MANIFEST_corpus_sft_v2.md`）。
> 无 manifest 的数据集不得用于训练（见 `.planning/REQUIREMENTS.md` DATA-11）。

## 基本信息

- **数据集版本**：<如 corpus_sft_v2>
- **生成日期**：<YYYY-MM-DD>
- **对应训练 run**：<run 名 + notebook/script，如 04_qlora_finetune / scripts/train_qlora.py>

## 生成脚本与参数

- **脚本**：<如 utils/corpus_to_sft.py，附 commit SHA>
- **关键参数**：<如 min_output_chars、dedup、目标 token 数、子集配额、multipliers>
- **随机种子 / API 模型版本**：<如 Moonshot 模型版本、temperature>

## 语料来源与哈希

- **来源语料**：<如 TRIZ-raw/，文件数，版本说明>
- **语料哈希**：<SHA-256（语料目录或 triz_corpus.jsonl）>

## Split 条数

| Split | 条数 |
|---|---|
| train | <n> |
| val | <n> |
| test | <n> |

## 子集分布

| 子集 | 条数 | 占比 | 配额目标 | 是否达标 |
|---|---|---|---|---|
| concept_explanation | | | | |
| contradiction_analysis | | | | |
| principle_recommendation | | | | |
| case_generation | | | | |
| ariz_guidance | | | | |
| innovation_assessment | | | | |
| safety_refusal | | | | |

## 质量门统计

- **去重**：移除 <n> 条（完全重复 <n> / 近似重复 <n>）
- **长度过滤**：移除 <n> 条（阈值 <>）
- **think 块清洗**：清洗/移除 <n> 条
- **困惑度过滤**：剔除分布尾部 <%>（如启用）
- **同题不同答冲突检查**：<n> 组，处理方式 <>

## 人工抽检

- **抽检比例**：<%（V2 质量门要求 ≥2%）>
- **抽检条数 / 通过条数 / 通过率**：<n / n / %>
- **抽检人与日期**：<>

## 备注

<已知缺陷、与上一版的 diff 摘要>
