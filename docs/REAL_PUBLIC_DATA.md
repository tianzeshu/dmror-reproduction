# DM-ROR 真实网络来源数据工程（2026-10-05）

本页的原始文件、归一化文件及完整核验命令描述 [v1.2.0 Release](https://github.com/tianzeshu/dmror-reproduction/releases/tag/v1.2.0) 工程。Git 的 `real_public/` 仅保留采集/处理代码、实际统计、来源和证据核验元数据；真实归一化 gzip 在 Release 的小型数据包或完整工程分卷中。Git 范围和核验见 [README_GIT_SUBSET.md](../real_public/README_GIT_SUBSET.md)。

本交付只包含实际从官方公开网络来源下载的记录及其归一化投影，没有生成企业、文本、关系、风险事件、负标签或传播路径来凑数量。原始 SC-Auto、SC-Semi、SC-Energy 仍未找回，原论文 Table 1 不能用这些新资料直接宣称复现。此前 v1.1.0 的三套 Synthetic 数据和 60 次实验属于历史模拟工程，不是本交付的真实数据或结果。

## 实际获得的资料

| 来源 | 实际数据 | 期间和统计口径 |
|---|---|---|
| NHTSA 汽车安全召回 | 185,699 原始记录行、8,090 去重 campaign；899 个报告制造商名称、22,547 个类型/品牌/型号/年款产品键；32,917 图节点、88,959 事实边 | RCDATE 2018–2025；包含车辆、设备、轮胎及儿童座椅。重复车型行不计作独立文章 |
| NHTSA vPIC | 10,000 个官方制造商 Mfr_ID | 2026 当前登记库分页 1–100，部分覆盖；不与召回名称自动合并，不充当历史企业集合 |
| EIA-860 能源 | 456,017 年度来源行；10,850 登记实体、18,215 厂站、38,521 机组、59 地域代码、85 NAICS 代码、38 燃料代码；203,171 去重事实边 | 全部八个 2018–2025 年度原始 ZIP。厂站和机组按物理资产记录，不冒称论文产品 |
| DOE-417 电力事故 | 2,006 官方事件，其中 32 条开始日期为空并保留 | 2018–2023；2024、2025 尚未获得。区域事故不能自动标为区域内所有企业的风险 |
| Wikidata 工业实体 | 6,524 实体、24,032 两端完整事实、24,166 来源 statement；保留 references / qualifiers | 2026 当前快照，汽车/半导体/能源候选可重叠。严格类型：企业 1,183、产品 114、材料 106、产业 169、地域 1,323、未分类 3,629 |

统计 CSV 为 `DATASET_STATISTICS_REAL.csv`，来源、许可和覆盖为 `SOURCE_CATALOG.json`。不同来源和类型不可随意相加当作原表 Firms/Products。Wikidata 已知类型链支持的半导体候选企业为 120；未知类型及未采目标保留为未知，不填造企业。产业候选标记也不等同于经过人工确认的企业部门分类。

## 文件和复核

`nhtsa_sources/` 保存召回、产品及制造商事实；`energy_sources/` 保存年度实体/关系及事故；`graph_sources/` 保存官方实体响应、类型证据和断言。原始 ZIP / XLS / API JSON 与来源 URL、UTC 下载时间、SHA-256 保留。`normalized/*.jsonl.gz`、`*.csv.gz` 是原始归一化字节的无损 gzip，解压后不会生成新数据。

在本目录运行（校验、解压仅需 Python 标准库）：

```bash
python verify_real_public_delivery.py --root .
python materialize_real_public_data.py --root .
python -m pip install -r requirements.txt
```

第二条命令恢复所有 JSONL/CSV，并从保存的官方 ZIP 复原 NHTSA TXT，逐文件核对大小和 SHA-256；不同内容的已有文件会被拒绝覆盖。解压需要足够磁盘空间。可以直接用 `gzip.open(path, "rt", encoding="utf-8")` 流式读取压缩 JSONL，不必全部展开。

完整 Release 包包含原始来源及归一化文件；Git 仓库精简目录只包含采集/处理代码、实际统计、来源和证据核验元数据。全包校验器需要完整原始文件清单，请对 Release 完整工程包运行。各来源内 README 及采集/归一化脚本可用于从快照复核；再次向网络下载时，官方数据更新可能产生不同数量和摘要，必须另记新采集版本。

NHTSA 全 185,699 行的29字段、8,090 事件和88,959边已回查官方原字节；EIA 全 456,017 行及7,185,754字段、全部实体/关系和DOE2,006事件已回查原Excel；Wikidata实体/类型/statement已逐条核对保存官方响应。来源支持表示与发布机构记录一致，不是对社区事实或因果解释的人工认证。

## 接入模型所需的真实证据

本包没有冒充可直接复现原论文的 `dataset.npz` 或新成绩。官方召回是实际产品安全事件，不等同于供应链停止交付；DOE主要提供受影响区域，没有确定每家企业；Wikidata没有风险结局或供应中断金标准。未观测到召回或新闻，不自动变成真实负例。所有未知风险、韧性/库存缓冲和传播路径维持未知。

采集时间是2026；NHTSA历史叙述可能后来修订，EIA调查年份不是精确发布日期，Wikidata当前断言不能充当2018年已知先验。原论文预测实验需要先核定实际事件定义、实体映射、正负标签的证据覆盖、供应依赖及历史可见时间，再构建时间划分和重新训练。`TRAINING_READINESS.json` 列出未完成条件，已有模型的人工证据接入规范见公开仓库 `docs/ANNOTATION_GUIDE.md`。本次没有把旧模拟实验成绩移到真实来源上。

## 来源和使用条件

- [NHTSA Datasets and APIs](https://www.nhtsa.gov/nhtsa-datasets-and-apis)；[Data.gov召回许可元数据](https://catalog.data.gov/dataset/nhtsas-office-of-defects-investigation-odi-recalls-nhtsa-api-6e97f)说明召回库的公共领域依据。vPIC 的[官方目录](https://catalog.data.gov/dataset/nhtsa-product-information-catalog-and-vehicle-listing-vpic-vehicle-api-json)说明数据可供公众自由使用，但 formal license 元数据是 unknown-license，未指定 SPDX 许可；不把召回库的许可外推到登记库。保留机构归属及采集版本。
- [EIA-860年度调查](https://www.eia.gov/electricity/data/eia860/)；[EIA公开使用规则](https://www.eia.gov/about/copyrights_reuse.php)。政府信息产品在公共领域，引用原机构；不要擅自扩大到第三方图片或商标。
- [DOE-417年度事故](https://doe417.pnnl.gov/)：引用原年度报告及文件URL。
- [Wikidata数据访问](https://www.wikidata.org/wiki/Wikidata:Data_access)：数据CC0-1.0，保留QID、statement、references与快照日期。

ETO/CSET半导体静态数据仅列候选链接，因其使用条款限制自动采集，本次未下载；IC-SPLC的公开页面/元数据在当前网络未成功取得，也不计入已采数据。此前恢复的新闻正文及混合模拟缓存不在这个公开真实数据包中。新增采集/处理代码沿用工程MIT许可，第三方数据按各自来源条件处理。
