# 数据构建工具 — 重建两个运行时 `.pkl`

两个快照运行时各依赖 `Data/` 下的两个大文件。它们**不进 git**(单个 >140 MB，超 GitHub 100 MB 限制），改为**用本目录脚本从公开数据源重建**。两个快照的 `.pkl` 内容一致，构建一次即可拷给两边。

| 产物 | 大小 | 作用 | 构建脚本 |
|------|------|------|----------|
| `Data/uniprot_kmer_index.pkl` | ~146 MB | 离线靶点识别：把服务器给的蛋白序列用 6-mer 反查 UniProt，得到 gene/UniProt ID | `build_uniprot_index.py` |
| `Data/pocket_db.pkl` | ~141 MB | 离线口袋库：每个药物靶点预存若干 holo 结构的 CA 坐标 + 配体质心，运行时 Kabsch 对齐取共识口袋中心 | `build_pocket_db.py` |

---

## 1. `uniprot_kmer_index.pkl` — UniProt k-mer 索引

**输入**：UniProt human Swiss-Prot FASTA（`UP000005640` 人类参考蛋白组，reviewed）。
从 https://www.uniprot.org/ 下载（Proteomes → *Homo sapiens* → reviewed → FASTA），约 2 万条。

**构建**：

```bash
python3 build_uniprot_index.py human_swissprot.fasta \
    ../submission_fornax/Data/uniprot_kmer_index.pkl
```

- 6-mer 全序列索引；跳过 <80 aa 的片段；剔除出现在 >50 个蛋白里的过常见 6-mer（不具区分度）。
- 产物结构：`{"kmer_size": 6, "entries": {uid: {gene, desc, seq_len}}, "kmer_to_ids": {kmer: [uid,...]}}`
- 我们构建时得到 **20036 条蛋白**。几分钟即可完成。

## 2. `pocket_db.pkl` — 离线口袋库

**输入**：靶点 UniProt 列表 `target_uniprots.txt`（**已随仓提供**，1819 个，即我们实际构建的全部靶点；来自 ChEMBL 活性数据里按活性数排序的 top 靶点）。

**构建**（联网直连 RCSB / UniProt，多线程）：

```bash
python3 build_pocket_db.py \
    --pocket-finder ../submission_fornax/Code \
    --out ../submission_fornax/Data/pocket_db.pkl \
    --workers 16
```

- 对每个 UniProt：RCSB 序列检索(identity 0.5, 深扫 80 个结构) → 逐个下载 PDB → 只在**序列匹配该靶点**的链上找配体 → 要求配体**够大**(≥12 原子)、**非结晶伪影**(排除 NAG 糖 / MSE / 缓冲剂等黑名单)、**埋得够深**(12 Å 内 ≥15 个 CA 接触，滤掉表面配体)。每靶保留最多 6 个 holo 参考。
- 这套"同链 + 药物样 + 埋深"三重过滤，正是为了避开 *Thrombin=NAG* 那类**错口袋**（历史踩过的坑）。
- 断点续跑：`--out` 已存在则跳过已完成靶点；每 25 个自动落盘。
- 全量 1819 靶 ≈ 1–2 小时(16 线程，网络受限)。产物 ~141 MB。
- 只重建部分靶点：`--uniprots P29597,P24941`

> `--pocket-finder` 指向任一快照的 `Code/`（两者共用同一份 `pocket_finder.py`，脚本从中复用 PDB 解析 / 配体黑名单 / 埋深判据等 helper）。

---

## 构建完成后

把生成的两个 `.pkl` 放进**各自快照**的 `Data/` 目录：

```
submission_fornax/Data/uniprot_kmer_index.pkl
submission_fornax/Data/pocket_db.pkl
submission_0771_snapshot_20260624/Data/uniprot_kmer_index.pkl
submission_0771_snapshot_20260624/Data/pocket_db.pkl
```

两快照内容一致，直接拷贝复用即可。之后按各快照的 `Dockerfile` / `Code/run.sh` 运行。
