# DM-ROR：代码、冻结数据与实测结果

根据 DM-ROR 方法和框架图实现的研究工程：双记忆、时间注意力、门控融合、节点韧性阈值、关系缓冲、残余过载传播及路径搜索。提供训练、评估、消融、预测和绘图代码。

**原稿 SC-Auto、SC-Semi、SC-Energy 的原始数据及 Table 1–5 实验记录尚未找回。** 本工程包含三个明确命名的合成 SIM 数据集和一个工业文本弱标签 ICKG-Weak 数据集；报告全部来自本工程实际执行的实验，不能当作原稿表格的数值复现。

## 获取工程和完整实验输出

Git 仓库包含代码、四套冻结 `dataset.npz`、数值 LLM 特征缓存、训练协议、报告、图和预测示例。全部 **260 次正式实验及 64 次验证搜索** 的权重、逐样本预测、配置、训练历史和日志位于 [v1.0.0 Release](https://github.com/tianzeshu/dmror-reproduction/releases/tag/v1.0.0) 的 `DMROR_Public_Delivery_20261005.zip`；解压到独立目录后运行下方严格验收命令。

公开版不包含新闻正文/证据片段、8B 模型权重、未发表论文全文、数据库历史或设备文件目录。来源 URL、日期、文档和片段 ID 保留于 `data/source_metadata/`，表格核查摘要位于 `recovery/`。冻结 NPZ 可直接训练和评估；从新闻原文完整重建 ICKG 需要另行取得合法来源文件，详见 [公开范围](docs/PUBLIC_RELEASE.md) 和 [重建条件](docs/REBUILD_DATA.md)。

## 数据集

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
| [人工标注指南](docs/ANNOTATION_GUIDE.md) | 真实标签导入及未知标签处理 |
| [研究缺口](docs/RESEARCH_GAPS.md) | 原稿结论缺失的证据和基线 |
| [代码验收](docs/HANDOFF_CHECKS.md) | 严格验证内容及历史限制 |
| [Table 1–5 核查](recovery/README.md) | 原稿转录与相关来源清单 |

正式训练的 14 个源文件保留于 `provenance/frozen_training_source/dmror`。早期 16 次完整模型验证搜索未记录源码指纹，原记录保留；后续 48 次基线搜索及全部正式运行有指纹。训练后缓存回放工具的跨平台修正和新增工具在 `docs/POST_TRAINING_CHANGES.json` 披露。

MIT 许可证适用于本工程新增代码及相应说明。新闻、模型和原稿资料的来源权利不因本仓库公开而改变；数值研究数据、来源元数据和转录表仅用于核查与复现，来源及局限见 [PUBLIC_RELEASE.md](docs/PUBLIC_RELEASE.md)。
