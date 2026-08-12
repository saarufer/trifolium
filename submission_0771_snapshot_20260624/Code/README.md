# Task2 复赛 — 自主分子设计 Agent（提交说明）

本镜像对 3 个蛋白靶点（`/saisdata/37/target{1,2,3}.pdb`）各自**实时运行**一个自主设计
Agent，为每个靶点产出一个分子 SMILES + 合成路线，写入 `/saisresult/result.zip`
（`result1/2/3.csv`，每个两列 `mol_smiles,route`）。

**所有结果均由 Agent 在容器内实时生成 —— 不预置、不复制、不从任何自带分子库筛选/抽取。**
内置的只有通用类药「起始种子」（`generate.load_seeds`，普通药物片段），最终分子由 Agent
的 设计→生成→改进→演化 循环产生。

---

## 1. 官方运行入口

入口固定为 `/app/run.sh`：
1. `timeout` 包裹 `python3 /app/Code/main.py`（外层硬超时，保证按时打包退出）；
2. 若 `result.zip` 缺失，触发 `--emergency-deliver` 兜底打包；
3. 全程日志写 `/app/run.log`，审计留痕（pocket 来源 / 对接 / 每轮冠军）随 csv 一起进 zip。

## 2. 输入与输出

- **输入**：`/saisdata/37/target1.pdb`、`target2.pdb`、`target3.pdb`
  （`find_targets()` 按 `targetN.pdb` 提取靶号，并有 `*.pdb` 通配兜底；B 榜换靶亦适配）。
- **输出**：`/saisresult/result.zip` 内含 `result1.csv`、`result2.csv`、`result3.csv`
  （在 `/app` 内构建后以 `copyfileobj` 写出，规避 `/saisresult` 不支持 seek 的限制）。

## 3. Agent 工作流程（设计→生成→改进→演化）

每个靶点独立运行 `Agent.run()`（见 `agent.py`），是一个感知-决策-行动-反思循环：

1. **靶点感知** `pocket_finder`：离线同源库（UniProt 键）定位真实结合口袋，抗残基重编号
   + 去结晶污染；ECS 数据网关兜底。
2. **起始种群** `generate.load_seeds`：内置通用类药片段作为**进化起点**（非结果）。
3. **决策** `brain.RuleBrain.decide(obs)`：根据种群停滞/多样性观测，选择下一步算子。
4. **生成/改进** `generate` 算子：`mutate`（定向突变）、`annulate`（成环/稠合）、
   `cross`（BRICS 互补杂交）——逐代演化出更优结合的新分子。
5. **打分** `objective` + `docking`：AutoDock Vina（镜像内 `/app/vina`，subprocess 调用，
   Open Babel 转 PDBQT）真实对接打分；嵌入稳定性门过滤难嵌入假阳。
6. **路线** `route.py`：命名反应逆合成（酰胺偶联 / 磺酰胺 / 酯 / 醚），输出含 `>>` 的
   合法反应式（规避 route 归零雷）。
7. **择优** `objective.select_champion`：可合成优先，从全代种群选最终冠军。

## 4. 依赖环境

- Python 3.11，`rdkit==2024.3.2`、`numpy==1.26.4`（化学与打分）。
- AutoDock Vina（Linux x86_64 静态二进制，烤入 `/app/vina`）+ Open Babel（PDBQT 转换）。
- `anthropic` / `httpx`：可选 LLM 战略层依赖（本提交以纯规则模式运行，见第 5 节）。

## 5. APIKEY 说明

镜像通过 `ANTHROPIC_API_KEY`（构建期注入）+ `ANTHROPIC_BASE_URL`（新加坡 ECS 代理）支持
一个**可选**的 LLM 战略层（`agent.py` 的 `_consult_strategist`，每 N 轮诊断并注入定向候选）。
**本提交以纯规则模式运行**（`strategist=None`）：规则 GA 循环独立完成全部设计→演化，
LLM 不参与，保证产出可复现、不依赖外部网络。APIKEY 仅为接口完整性保留。

## 6. 复现与合规

- 每靶固定预算（`TARGET_BUDGET_SEC`），到点择优打包，超时由 run.sh 外层兜底。
- result.log 随结果进 zip，记录每靶 pocket 来源、对接分、各轮冠军轨迹 —— 可审计 Agent
  确实**实时运行并演化**，而非读取预置答案。
