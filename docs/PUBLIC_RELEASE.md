# 公开发布范围与重建边界

公开日期：2026-10-05。仓库：`tianzeshu/dmror-reproduction`。Table 1 完整规模合成数据及 60 次新实验输出：[v1.1.0 Release](https://github.com/tianzeshu/dmror-reproduction/releases/tag/v1.1.0)。历史小型 SIM/ICKG 实验输出：[v1.0.0 Release](https://github.com/tianzeshu/dmror-reproduction/releases/tag/v1.0.0)。

## 历史 v1.0.0 工程下载与合并

完整工程以五个无损分卷发布：下载 `DMROR_Public_Delivery_20261005.zip.part01` 至 `.part05` 和 `DMROR_Public_Delivery_20261005.parts.sha256`，将五个分卷放在同一目录。使用仓库脚本合并并逐分卷核验（只需 Python 标准库）：

```bash
python scripts/assemble_release.py --parts-dir /path/to/downloads --output-dir /path/to/downloads
```

脚本固定了五个分卷的 SHA-256，不会覆盖不同内容的已有文件；已有正确 ZIP 仅校验。恢复后的文件名仍为 `DMROR_Public_Delivery_20261005.zip`，完整 SHA-256 为 `3cb88e7e65bc519a547af2ec7ec95c58834cce711de0f83fe53538c726a6ca78`。解压后再执行工程命令。

## 公开内容

- DM-ROR 代码、脚本、测试、方法说明、协议、原训练源码快照。
- SIM-Auto、SIM-Semi、SIM-Energy 和 ICKG-Weak 的原冻结数值 NPZ、metadata、观察索引；三套 SIM 的自建模板信号。
- `embedding_cache.npz` 原数值向量及固定随机投影；没有新闻字符串或模型权重。
- 本工程实测报告、图、预测示例。完整 Release 另含 260 次正式实验与 64 次验证搜索的全部权重、配置、预测、历史及日志。
- 原材料来源 URL、文档/片段 ID、日期、字符串省略后的信号与标签证据元数据。相关恢复新闻仅公开 10,010 条 URL/日期等来源元数据。
- Table 1–5 原稿数字转录与核查摘要；转录表不是训练数据或实测输出。

所有冻结 `dataset.npz`、`embedding_cache.npz`、训练源码和保留的历史实验文件均从原交付逐字节复制，原 SHA-256 不变。公开说明及 JSONL 元数据是本次新增或改写，不替代原证据。

## 省略内容

新闻正文、文本片段、原文标签支持段落、包含原文的 LLM 输入/输出、`data/raw/ickg/`、`data/llm_cache/texts.jsonl`、ICKG `signals.jsonl` 和 `label_evidence.jsonl` 均未公开。未发表论文 TeX/PDF、Overleaf/Mongo 项目历史、聊天、设备检索清单和原内部交付 ZIP 均未公开。没有公开 8B 模型权重。

原新闻正文的逐来源再分发许可没有完成核验，因此本公开工程使用数值冻结输入及去正文的元数据。数据集中仍保留来源 ID 和 URL，便于在有合法访问条件时核查。

## 能复现什么

冻结 NPZ 已包含图、输入向量、观察索引、时间划分和标签，可以直接训练、评估和复算已有结果，不需要原文。Release 完整工程可运行原严格 `verify_delivery.py`；Git 克隆目录因省略 results，不能单独验收全部历史任务。

合成场景可由 `build_simulated` 重新生成，其初始向量是诊断 hash 特征。原冻结 LLM 特征直接使用已公开 NPZ。原 `apply_cache` 按完全相同的文本匹配 `texts.jsonl`；公开版省略这个原文清单，因此不能仅凭公开副本运行该回放命令并声称已完整重建原 LLM 数据。添加新文本需合法来源及另一个编码版本。

ICKG 从原文重建需要自行合法取得对应 contexts、实体、关系和链接四文件，核对 `docs/REBUILD_DATA.md` 中源摘要，才能运行 `build_ickg`。URL 列表不等于已取得原文，当前网页也不保证与历史快照逐字相同。不得绕过摘要校验或依据原稿 Table 1 数量补造数据。

## 许可证和解释范围

`LICENSE` 的 MIT 条款适用于本工程新增代码及其说明，不能用于重新授权第三方新闻、模型权重或原稿。来源资料仍受原提供方权限和条款约束。

SIM 是合成实验；ICKG-Weak 是未来词汇报道代理任务。真实节点中断、库存/容量/韧性、因果风险源及传播路径金标准尚缺，不能将本工程当作论文 SC 三域原始数据或 Table 1–5 实测结果的恢复。
# v1.1.0：Table 1 规模合成数据

新增 `data/table_scale/SC-Auto-Synthetic`、`SC-Semi-Synthetic` 和 `SC-Energy-Synthetic`。企业、产品、材料、行业、地域、关系、文本、风险信号和带标签实例数量按用户给出的 Table 1 构建。它们含有原创合成文本，允许随本工程公开交付；它们不是原稿真实 SC 数据，也不包含未公开论文全文或第三方新闻正文。

新规模训练与历史 v1.0.0 分开记录。结果配置保留对应 NPZ、运行配置和实现文件摘要；历史来源冻结目录及旧 Release 保留。图训练采用数学等价的低显存关系投影，路径检索复用同一天的邻接结构；这些调整的说明见 `TABLE_SCALE_CHANGES.json`。

每个公开新数据集提供原始合成记录、可训练 NPZ、元数据和独立数量/时间验收报告。生成与计数定义见 `DATASET_TABLE1_SCALE.md`。哈希编码是明确披露的诊断特征，不能描述为已执行完整 LLM 提示编码。



## v1.2.0 真实公开来源增补

`real_public/` 是独立版本的真实来源采集代码与元数据，不属于 SIM、ICKG-Weak 或 Table 1 Synthetic。完整 v1.2.0 Release 包包含官方 NHTSA、EIA/DOE 数据文件、Wikidata CC0 API 响应和全部归一化数据；Release 另提供小型归一化数据包。Git 仅包含采集/处理代码、实际统计、来源和证据核验元数据，真实 JSONL/CSV gzip 不随 Git 克隆分发。数据按 [真实来源说明](REAL_PUBLIC_DATA.md) 中各自条件提供，工程 MIT 许可不替代第三方数据条件。没有加入旧新闻正文、未发表稿、模型权重或设备缓存。没有真实供应风险评估成绩；旧报告只对应其注明的历史数据。
