# DM-ROR：代码、冻结数据与实测结果

根据 DM-ROR 方法和框架图实现的研究工程：双记忆、时间注意力、门控融合、节点韧性阈值、关系缓冲、残余过载传播及路径搜索。提供训练、评估、消融、预测和绘图代码。

**原稿 SC-Auto、SC-Semi、SC-Energy 的原始数据及 Table 1–5 实验记录尚未找回。** 本工程新增按用户提供的 Table 1 逐项计数构建的三套完整规模合成数据，名称为 `SC-*-Synthetic`；同时保留历史 SIM 和 ICKG-Weak 数据。所有报告区分数据来源和版本，不能当作原稿真实数据实验的数值复现。

## Table 1 完整规模数据

完整规模代码、数据及 **60 次实际训练输出** 位于 [v1.1.0 Release](https://github.com/tianzeshu/dmror-reproduction/releases/tag/v1.1.0)。[中文实验报告](reports/table_scale/experiment_report_zh.md) 和 [逐种子数据](reports/table_scale/per_seed.csv) 对应 3 个数据集 × 4 个方法 × 5 个种子；代码测试 73 项全部通过。数据和实验分别有独立验收报告。

仅需数据时下载 `DMROR_Table1_Scale_Datasets_20261005.zip`。完整工程另含全部权重、预测、配置和日志，以 `DMROR_Table1_Scale_Delivery_20261005.zip.partXX` 分卷发布；下载全部分卷及 `table_scale_parts.json`、`assemble_table_scale.py` 到同一目录，运行 `python assemble_table_scale.py` 合并并校验 SHA-256，再解压。历史 v1.0.0 下的小型 SIM 实验另列于下方。

| 数据集 | 企业 | 产品 | 材料 | 行业 | 地域 | 总节点 | 关系 | 文本 | 风险信号 | 带标签实例 | 时间 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SC-Auto-Synthetic | 6,482 | 1,126 | 438 | 76 | 42 | 8,164 | 58,734 | 124,680 | 38,912 | 8,426 | 2018–2025 |
| SC-Semi-Synthetic | 4,935 | 842 | 316 | 54 | 37 | 6,184 | 46,218 | 96,440 | 31,705 | 6,913 | 2018–2025 |
| SC-Energy-Synthetic | 5,714 | 973 | 512 | 68 | 51 | 7,318 | 52,906 | 108,375 | 34,286 | 7,584 | 2018–2025 |

`data/table_scale/` 保存实体、关系、原创合成文本、信号、标注和可训练的 NPZ。`#Risk Labels` 按带已知 0/1 标签的实例总数统计，未标注实例为 `-1`。这三套数据按表格规模构建，属于合成数据；数量一致不代表找回了原始真实供应链数据。规则与计数口径见 [构建说明](docs/DATASET_TABLE1_SCALE.md)。

```bash
python scripts/verify_table_scale.py --data-root data/table_scale --output reports/table_scale/data_validation.json
python -m dmror.train --dataset data/table_scale/SC-Auto-Synthetic/dataset.npz --output results/table_scale/my_run --mode full --seed 17 --epochs 60 --hidden 128 --batch-size 1 --lr 0.0003 --device cuda:0
```

如需从头重建，使用新的目录，例如 `python -m dmror.build_table_scale --all --output data/rebuilt/table_scale`；生成器拒绝覆盖已有数据。无 CUDA 的环境使用 `--device cpu`。大图会比下方历史小型验证数据需要更多训练时间与内存。新规模实验使用独立输出目录，旧 SIM 报告保持原数据与原文件摘要。

## 获取历史 v1.0.0 工程和实验输出

Git 仓库包含代码、四套冻结 `dataset.npz`、数值 LLM 特征缓存、训练协议、报告、图和预测示例。全部 **260 次正式实验及 64 次验证搜索** 的权重、逐样本预测、配置、训练历史和日志位于 [v1.0.0 Release](https://github.com/tianzeshu/dmror-reproduction/releases/tag/v1.0.0) 的五个 ZIP 分卷；合并后解压到独立目录，运行下方严格验收命令。

完整工程以五个无损分卷发布：下载 `DMROR_Public_Delivery_20261005.zip.part01` 至 `.part05` 和 `DMROR_Public_Delivery_20261005.parts.sha256`，将五个分卷放在同一目录。使用仓库脚本合并并逐分卷核验（只需 Python 标准库）：

```bash
python scripts/assemble_release.py --parts-dir /path/to/downloads --output-dir /path/to/downloads
```

脚本固定了五个分卷的 SHA-256，不会覆盖不同内容的已有文件；已有正确 ZIP 仅校验。恢复后的文件名仍为 `DMROR_Public_Delivery_20261005.zip`，完整 SHA-256 为 `3cb88e7e65bc519a547af2ec7ec95c58834cce711de0f83fe53538c726a6ca78`。解压后再执行工程命令。


公开版不包含新闻正文/证据片段、8B 模型权重、未发表论文全文、数据库历史或设备文件目录。来源 URL、日期、文档和片段 ID 保留于 `data/source_metadata/`，表格核查摘要位于 `recovery/`。冻结 NPZ 可直接训练和评估；从新闻原文完整重建 ICKG 需要另行取得合法来源文件，详见 [公开范围](docs/PUBLIC_RELEASE.md) 和 [重建条件](docs/REBUILD_DATA.md)。

## 历史验证数据集

| 数据集 | 节点 | 边 | 查询次数 | 标签含义 |
|---|---:|---:|---:|---|
| SIM-Auto | 112 | 450 | 84 | 独立离散级联仿真 |
| SIM-Semi | 112 | 450 | 84 | 独立离散级联仿真 |
| SIM-Energy | 112 | 450 | 84 | 独立离散级联仿真 |
| ICKG-Weak | 191 | 197 | 131 | 未来 30 天风险词汇报道代理任务 |

ICKG-Weak 没有实测韧性和缓冲属性，相关观察掩码为 false；边和路径真值未知，对应路径指标为 NA。文本风险词汇不构成真实企业中断标注。

## 快速运行

```bash
python -m venv .venv
# 激活虚拟环境后执行；CPU/GPU PyTorch 安装详见 docs/QUICKSTART.md
python -m pip install -e ".[test]"
python -m pytest -q
python -m dmror.data data/prepared/SIM-Semi/dataset.npz
python -m dmror.train --dataset data/prepared/SIM-Semi/dataset.npz --output results/my_run --mode full --seed 17 --epochs 60 --device cpu
python -m dmror.predict --run-dir results/my_run --dataset data/prepared/SIM-Semi/dataset.npz --output predictions/my_run --split test
```

训练和预测无需新闻原文或重新加载 8B 模型。GPU 环境将 `--device cpu` 改为 `--device cuda:0`。正式协议包含 4 个数据集 × 13 个模式 × 5 个种子（17、29、43、71、101），验证 AUPRC 选择 checkpoint，分类阈值只由验证集确定。三个独立工程基线为 text_mlp、graph_only、concat；原稿 BERT、FinBERT、HGT、TGN 等命名基线尚需另行实现验证。

## 完整输出验收与复跑

在 Release 完整工程根目录执行：

```bash
python scripts/verify_delivery.py --frozen-source provenance/frozen_training_source/dmror --output reports/user_delivery_validation.json
```

严格检查 260 次正式运行、64 次验证搜索、52 组五种子结果、输入摘要、历史源码、指标和阈值。Git 克隆目录未包含这些大规模实验输出，直接运行完整验收会报告缺失；请使用 Release 工程。原验收程序保持不变，不需新闻正文，也不会加载权重 pickle。

复跑时复制 `protocols/formal.json` 并改为新输出目录，再执行 `scripts/run_parallel.py`；具体命令见 [运行指南](docs/QUICKSTART.md)。报告中的种子标准差描述训练随机性，不是总体置信区间；预测窗口重叠存在时间相关性。

## 文档

| 文件 | 用途 |
|---|---|
| [快速运行](docs/QUICKSTART.md) | 环境、训练、预测与正式复跑 |
| [实测实验报告](reports/experiment_report_zh.md) | 五种子结果、有效指标和局限 |
| [方法对应说明](docs/METHOD_MAPPING.md) | 公式、框架与具体工程实现 |
| [数据规范](docs/DATA_CONTRACT.md) | 张量、时间边界、弱标签及缺失观测 |
| [公开版重建条件](docs/REBUILD_DATA.md) | 合成数据与合法原文重建流程 |
| [Table 1 完整规模构建](docs/DATASET_TABLE1_SCALE.md) | 新三套数据的统计定义、生成和验收 |
| [新规模实测实验报告](reports/table_scale/experiment_report_zh.md) | 60 次新训练的五种子结果、配置与局限 |
| [人工标注指南](docs/ANNOTATION_GUIDE.md) | 真实标签导入及未知标签处理 |
| [研究缺口](docs/RESEARCH_GAPS.md) | 原稿结论缺失的证据和基线 |
| [代码验收](docs/HANDOFF_CHECKS.md) | 严格验证内容及历史限制 |
| [Table 1–5 核查](recovery/README.md) | 原稿转录与相关来源清单 |

正式训练的 14 个源文件保留于 `provenance/frozen_training_source/dmror`。早期 16 次完整模型验证搜索未记录源码指纹，原记录保留；后续 48 次基线搜索及全部正式运行有指纹。训练后缓存回放工具的跨平台修正和新增工具在 `docs/POST_TRAINING_CHANGES.json` 披露。

MIT 许可证适用于本工程新增代码及相应说明。新闻、模型和原稿资料的来源权利不因本仓库公开而改变；数值研究数据、来源元数据和转录表仅用于核查与复现，来源及局限见 [PUBLIC_RELEASE.md](docs/PUBLIC_RELEASE.md)。
