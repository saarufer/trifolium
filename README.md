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

本仓库收录两份**效果最好**的提交：

| 目录 | 平台分 | 说明 |
|------|:------:|--------|
| [`submission_fornax/`](submission_fornax/) | **0.775436** 🥇 | Fornax 架构：规则开局 → LLM 精修 |
| [`submission_0771_snapshot_20260624/`](submission_0771_snapshot_20260624/) | **0.771534** | 稳定基线快照 |

---

## 🧬 方案架构

```
靶点 PDB
   │
   ├─ 离线靶点识别      protein_id.py   ← uniprot_kmer_index.pkl（6-mer 反查 UniProt）
   ├─ 离线口袋定位      pocket.py       ← pocket_db.pkl（holo 结构 Kabsch 对齐取共识口袋）
   ├─ 分子生成          generate.py     （规则骨架 + 取代基/环稠合）
   ├─ 真对接筛选        docking.py      （AutoDock Vina，并行）
   ├─ 目标打分          objective.py    （binding 甜点 −11~−12 kcal/mol）
   ├─ 逆合成路线        route.py        （routable-first 冠军选择 + 路线守恒）
   └─ 冠军交付          main.py         （补齐 3 CSV + 应急打包）
```

生成分子经四道**类药性门**把关：柔性门（旋转键 ≤6）、greasy 门（长脂链无环）、稠环门、口袋埋深门，倾向清爽、可合成的分子。

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
<sub>感知 · 决策 · 行动 · 反思 —— Vina 真对接驱动的自治分子设计。</sub>
</div>
