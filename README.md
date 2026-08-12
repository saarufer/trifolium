<div align="center">

# 🍀 Trifolium — Task 2

**第四届世界科学智能大赛 · 复赛 Task 2（分子生成 + 逆合成路线）**

一个 **感知 → 决策 → 行动 → 反思** 的自治分子设计 agent：<br>
给定靶点 PDB，自动识别口袋，用 AutoDock Vina 真实对接驱动，设计**深结合 + 可合成 + 路线可行**的分子。

<br>

![Rank](https://img.shields.io/badge/复赛-第%207%20名-gold?style=for-the-badge)
![Best Score](https://img.shields.io/badge/最高分-0.7754-2ea44f?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![Docking](https://img.shields.io/badge/AutoDock%20Vina-1.2.x-005571?style=flat-square)
![Approach](https://img.shields.io/badge/approach-rules%20%2B%20optional%20LLM-8A2BE2?style=flat-square)

</div>

---

## 🏆 成绩

> **复赛第 7 名 · 最高平台分 0.775436**

本仓库收录两份**效果最好**的提交，均为**纯规则**方案（对照实验证明纯规则在该平台战力最强）：

| 目录 | 平台分 | 一句话 |
|------|:------:|--------|
| [`submission_fornax/`](submission_fornax/) | **0.775436** 🥇 | Fornax 架构：规则开局 → LLM 精修门。实测 LLM 因 warmup 门槛未介入，即纯规则真实战力 |
| [`submission_0771_snapshot_20260624/`](submission_0771_snapshot_20260624/) | **0.771534** | 历史新高基线：`select_champion` routable-first + route 三件套修复后冲上的稳定基线 |

---

## 💡 核心发现：**平台 binding ≠ 纯 Vina 深度**

整个赛程最关键、也最反直觉的一条结论 —— 靠对照实验（干净单变量）逼出来的：

| 版本 | 做了什么 | 平台分 | 结论 |
|------|----------|:------:|------|
| **fornax**（纯规则，清爽类药分子） | builtin 浅分子 | **0.7754** ✅ | 最优 |
| aggressive | 深种子 + 多重启 + 狂稠环，三靶 Vina **全部更深**（T1 −14.35） | 0.5525 ❌❌ | 暴跌 0.22 |
| lynx | LLM 真密集介入（+29K token）+ S1–S5 育种 | 0.6275 ❌ | 反跌 0.148 |

- **追深 Vina 是负优化**：aggressive 三靶 Vina 全更深，分数却暴跌 —— 平台能识破油腻大稠环假阳（冠军 logP 8.1 / 8 个芳环，consensus 稳定但"假深"）。
- **LLM 介入也拖后腿**：干净对照下，LLM 没介入（fornax 0.7754）> LLM 真介入（lynx 0.6275）。平台只认**清爽、类药、可合成**的分子。
- 最优策略：**纯规则 + Vina 真对接筛选 + 强逆合成路线守恒**，不刷深度、不堆稠环。

---

## 🧬 方案架构

```
靶点 PDB
   │
   ├─ 离线靶点识别      protein_id.py   ← uniprot_kmer_index.pkl（6-mer 反查 UniProt）
   ├─ 离线口袋定位      pocket.py       ← pocket_db.pkl（holo 结构 Kabsch 对齐取共识口袋）
   ├─ 分子生成          generate.py     （规则骨架 + 取代基/环稠合）
   ├─ 真对接筛选        docking.py      （AutoDock Vina，并行）
   ├─ 目标打分          objective.py    （对齐平台 binding 倒-U 峰 −11~−12）
   ├─ 逆合成路线        route.py        （routable-first 冠军选择 + 路线守恒）
   └─ 冠军交付          main.py         （补齐 3 CSV + 应急打包防漏交）
```

四道**防假阳门**（针对 Vina 的四类系统性假阳性）：柔性门（旋转键 ≤6）、greasy 门（长脂链无环）、cage 门（假深稠环）、埋深门。

---

## 📦 运行

```bash
cd submission_fornax          # 或 submission_0771_snapshot_20260624
docker build -t trifolium-task2 .
docker run --rm -v $PWD/out:/saisresult trifolium-task2
```

细节见各快照的 `Dockerfile` 与 `Code/run.sh`。

---

## 🔧 运行时数据（两个 `.pkl`）— 需先重建

每个快照运行时依赖 `Data/` 下两个大文件，因单文件 >140 MB 超 GitHub 100 MB 限制**未纳入 git**，改为**用脚本从公开数据源重建**：

| 文件 | 大小 | 作用 |
|------|:----:|------|
| `Data/uniprot_kmer_index.pkl` | ~146 MB | 离线靶点识别（UniProt 6-mer 索引，20036 蛋白）|
| `Data/pocket_db.pkl` | ~141 MB | 离线口袋库（1819 靶点的 holo 参考）|

重建方法、命令、耗时见 **[`tools/README.md`](tools/README.md)**。构建脚本 + 我们实际用的 1819 个靶点 UniProt 列表（`tools/target_uniprots.txt`）已随仓提供，`pocket_db` 无需任何本地大数据库即可复现。

```bash
cd tools
# 1) UniProt k-mer 索引（下载人类 Swiss-Prot FASTA 后）
python3 build_uniprot_index.py human_swissprot.fasta \
        ../submission_fornax/Data/uniprot_kmer_index.pkl
# 2) 口袋库（联网直连 RCSB/UniProt，~1-2h / 16 线程）
python3 build_pocket_db.py --pocket-finder ../submission_fornax/Code \
        --out ../submission_fornax/Data/pocket_db.pkl --workers 16
```

两快照的 `.pkl` 内容一致，构建一次拷给两边即可。

---

<div align="center">
<sub>纯规则 · Vina 真对接 · 路线守恒 —— 平台奖励清爽类药分子，而非假深稠环。</sub>
</div>
