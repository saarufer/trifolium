# trifolium — Task 2 最优提交

第四届世界科学智能大赛 复赛 Task 2（分子生成 + 逆合成路线）效果最好的两份提交代码。

均为**纯规则 autonomous agent** 方案（感知靶点 → 口袋识别 → 分子生成 → Vina 对接筛选 → 逆合成路线）。

| 目录 | 线上分 | 说明 |
|------|--------|------|
| `submission_0771_snapshot_20260624/` | **0.771534** | 历史基线快照，`select_champion` routable-first + route 三件套修复后冲上历史新高 |
| `submission_fornax/` | **0.775436** | Fornax 架构（规则开局 + LLM 精修门，实测 LLM 因 warmup 门槛过严未介入，等同纯规则战力） |

> 关键结论：平台 binding ≠ 纯 Vina 深度——追深 Vina（aggressive 版）暴跌到 0.55；LLM+育种（lynx 版）反跌到 0.63。最优仍是纯规则清爽类药分子。

## 未包含的数据文件

每个快照运行时依赖两个大数据文件，因超过 GitHub 100MB 单文件限制**未纳入本仓库**：

- `Data/uniprot_kmer_index.pkl`（~146MB）— UniProt k-mer 索引，用于离线靶点识别
- `Data/pocket_db.pkl`（~141MB）— 预构建口袋库

需从原始 `stage2` 工作区获取后放回各快照的 `Data/` 目录才能完整运行。

## 运行

见各快照 `Code/run.sh` 与 `Dockerfile`。
