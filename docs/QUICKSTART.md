# 学生接手运行指南

本工程提供 DM-ROR 的可运行实现、已处理数据、冻结文本特征、训练协议和实验输出。先直接使用 `data/prepared/` 跑通训练。完整历史权重和结果须从 [v1.0.0 Release](https://github.com/tianzeshu/dmror-reproduction/releases/tag/v1.0.0) 下载公开工程。原文重建的前提见 [REBUILD_DATA.md](REBUILD_DATA.md)，公开版没有新闻正文。全部命令在工程根目录执行。

交付数据的文本特征已经由本地冻结的 `Llama-3-8b-instruct` 编码为 64 维。训练和预测均不需要下载 8B 模型、不需要模型服务，也不需要访问 49 服务器。`data/llm_cache/` 包含 3,936 段唯一原文的向量、固定投影和编码来源记录。

## 1. 安装环境

使用 Python 3.10 或以上，建议单独创建虚拟环境：

```text
python -m venv .venv
```

Windows PowerShell 激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS 激活：

```bash
source .venv/bin/activate
```

如果 Windows 环境不允许激活脚本，后续命令中的 `python` 可直接替换为 `.\.venv\Scripts\python.exe`。

CPU 环境先安装 CPU 版 PyTorch，再安装工程：

```text
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
```

NVIDIA GPU 环境请先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 选择操作系统和适合驱动的 CUDA wheel，执行其安装命令；随后运行 `python -m pip install -e ".[test]"`。检查 GPU 是否可用：

```text
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

下文的 `--device cpu` 可改为 `--device cuda:0`。直接使用缓存数据时无需安装 `.[llm]`；该额外依赖只用于重新执行本地 LLM 编码或结构化提取。

## 2. 检查数据和代码

```text
python -m pytest -q
python -m dmror.data data/prepared/SIM-Auto/dataset.npz
python -m dmror.data data/prepared/ICKG-Weak/dataset.npz
```

`metadata.json` 记录数据定义、编码器、时间划分和 SHA-256。`docs/DATA_CONTRACT.md` 说明每个数组的含义；`docs/METHOD_MAPPING.md` 说明代码和论文方法的对应关系。

## 3. 先完成一个模型、一个种子

以下是短训练检查，最多 10 个 epoch；用独立目录保存结果。它用于确认环境和运行流程，不等同于完整五种子实验。

```text
python -m dmror.train --dataset data/prepared/SIM-Auto/dataset.npz --output results/my_first_run/SIM-Auto/full/seed_17 --mode full --seed 17 --epochs 10 --patience 5 --device cpu
```

输出包含 `checkpoint.pt`、训练集拟合的 `scaler.npz`、`config.json`、`history.json`、`result.json` 和保存的预测数组。只用验证 AUPRC 选择 checkpoint；分类阈值只由验证集确定。测试结果在 `result.json` 的 `test` 字段。

## 4. 按已冻结协议复跑正式实验

`protocols/formal.json` 保存本次验证选参后的最终参数：4 个数据集、13 种模型/消融、5 个训练种子（17、29、43、71、101），共 260 个训练任务。独立基线和完整模型各使用同样的四组验证搜索预算，消融沿用完整模型参数。要严格比较本次交付结果，应使用该协议。

先复制协议并指定新的输出目录，保留交付的 `results/formal/`：

```text
python -c "import json; from pathlib import Path; p=json.loads(Path('protocols/formal.json').read_text(encoding='utf-8')); p['output']='results/user_formal'; Path('protocols/user_formal.json').write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')"
python scripts/run_parallel.py --protocol protocols/user_formal.json --devices cpu
python -m dmror.report --results results/user_formal
```

一张 GPU 使用 `--devices cuda:0`，两张 GPU 可使用 `--devices cuda:0 cuda:1`。每个设备槽位同时运行一个任务。已存在的结果会触发保护检查；重新尝试时换一个输出目录，并保留失败的日志。260 个任务在 CPU 上会比较慢，可以先在新协议副本中缩小 `jobs`，检查一项正式配置，再执行全部任务。

如果想重新验证选参流程，请在新目录分别执行 `protocols/validation_search.json` 和 `protocols/baseline_validation_search.json`，再用 `scripts/select_validation.py` 冻结新协议。不要根据测试指标修改参数后继续称作同一次正式实验。硬件、PyTorch 版本和 GPU 算子差异可能影响训练浮点结果；数据文件可按 SHA-256 精确核对，训练指标按同协议多种子比较。

## 5. 从 checkpoint 导出风险节点和候选路径

使用上面的短训练结果：

```text
python -m dmror.predict --run-dir results/my_first_run/SIM-Auto/full/seed_17 --dataset data/prepared/SIM-Auto/dataset.npz --output results/my_first_predictions --device cpu --split test
```

`risk_predictions.json` 保存每个快照的风险节点、阈值、范围和有向候选路径；`predictions.npz` 保存节点概率、边概率和边强度。使用正式 checkpoint 时，将 `--run-dir` 改为 `results/formal/SIM-Auto/full/seed_17`。输入数据必须使用与训练相同的编码器和特征维度。新快照可以使用 `--split all`，无需提供标签；数组契约见 `DATA_CONTRACT.md`。

## 6. 解释结果时确认数据来源

| 数据集 | 实际来源与标签 | 可以检验的范围 |
|---|---|---|
| SIM-Auto / SIM-Semi / SIM-Energy | 固定种子的合成图、模板文本、容量和独立随机传播标签 | 代码训练、消融、模拟传播路径与范围评估 |
| ICKG-Weak | 49 服务器既有公开文本和自动关系；未来 30 天风险短语报道为弱标签 | 回顾式公开报道预测 |

尚未找到原稿 SC-Auto、SC-Semi、SC-Energy 对应的原始数据工程；交付的三个 SIM 数据集不是这三套原始数据。ICKG 资料中没有实测韧性属性、真实中断、因果传播路径或真实受影响集合，相关输入保留缺失 mask，路径真值为未知。因此这些结果可以说明当前实现的运行和上述任务表现，不能直接当作原稿全部数值的复现，也不能把报道预测准确率等同于真实供应链中断预测准确率。完整限制见 `DATA_CONTRACT.md` 和 `RESEARCH_GAPS.md`。

五个种子的均值和标准差描述训练随机性；数据图和时段固定、30 天预测窗口重叠，种子标准差不是总体置信区间。
# 按 Table 1 规模运行

完整规模数据位于 `data/table_scale/SC-Auto-Synthetic`、`SC-Semi-Synthetic`、`SC-Energy-Synthetic`。总节点数为 8,164、6,184、7,318；其余表格计数逐项一致。数据及文本均为明确标记的合成记录，原始真实 SC 数据尚未找回。

```bash
python scripts/verify_table_scale.py --data-root data/table_scale --output reports/table_scale/data_validation.json
python -m dmror.train --dataset data/table_scale/SC-Auto-Synthetic/dataset.npz --output results/table_scale/my_run --mode full --seed 17 --hidden 128 --batch-size 1 --lr 0.0003 --epochs 60 --device cuda:0
```

已有数据可直接验收和训练。从头重建时必须选择新目录，例如 `python -m dmror.build_table_scale --all --output data/rebuilt/table_scale`，生成器不会覆盖已存在的 `dataset.npz`。无 CUDA 的环境将 `--device cuda:0` 改成 `--device cpu`。

固定比较协议为 3 个数据集 × 完整模型和 3 个工程基线 × 5 个种子，共 60 次独立运行。正式结果使用 `results/table_scale`，数据摘要、配置、源码摘要、逐样本概率、权重和日志保存在每个运行目录。若复跑，复制协议并把 JSON 中的 `output` 改为新的结果目录，然后执行：

```bash
python scripts/run_parallel.py --protocol protocols/my_table_scale.json --devices cuda:0 --slots-per-device 1
python scripts/verify_table_scale_runs.py --protocol protocols/my_table_scale.json --results results/my_table_scale --output reports/my_table_scale_validation.json
```

`--slots-per-device` 默认 1；本机已使用 3 个并发任务，但应按实际可用显存和系统内存选择。多任务的运行时间包含资源竞争。单任务短程实测采用 RTX 5090 Laptop GPU、K=5、D=64、H=128、batch size=1，峰值 CUDA allocated 约 1.74 GiB，完整运行的数值见各自结果文件。

正式报告工具 `scripts/report_table_scale.py` 只接受完整的 60 次规定运行，拒绝混入短程 pilot、旧 SIM 或不同输入摘要的结果。规模、生成规则、有限标签与未知标签的区别见 `DATASET_TABLE1_SCALE.md`。

