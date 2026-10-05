# 真实公开 Wikidata 工业候选图

本目录是 2026 年 10 月 5 日通过官方接口实际采集的 Wikidata 快照，**不是模拟数据，不是原论文 SC 数据恢复，也不满足原论文表 1 的指定规模**。只保存实际官方响应中的实体和断言；不生成企业、产品、材料、关系、风险结局、供应关系或传播真值。

## 实际数量

| 内容 | 实际数量 |
|---|---:|
| 下载并保存的实体 | 6,524 |
| 排除明确虚构对象后的实体 | 6,524 |
| firm：有类型链支持的企业 | 1,183 |
| product：制造品或制造品类别 | 114 |
| material：材料、化学物质或元素 | 106 |
| industry：产业或经济部门 | 169 |
| region：国家、城市、行政区域或聚居地 | 1,323 |
| unclassified：类型链缺失或有冲突 | 3,629 |
| 可直接使用的闭合唯一事实边 | 24,032 |
| 支持上述事实边的原始 statements | 24,166 |
| 所有下载实体的完整 claims | 391,667 |
| 有观测结局的风险标签 | **0** |
| 有实际供应依赖证据的 Supplies 边 | **0** |

所有类型来自已保存的非弃用 `P31` / `P279` 断言链。企业类本身不当作企业实例计数。制造品、材料、产业中可包含类别概念，`entity_form = class_or_category` 明确标记；这 6,524 条中共有 928 条带 `P279` 的类别记录，不应全部当作企业实例。类型分类有深度和采集预算；未找到足够证据就留在 unclassified，不按名字或关系角色猜类型。

汽车、半导体、能源领域通过 `P452` 产业或直接 `P31` 类型查询得到候选，再保留与候选实际相连的一跳目标；领域可以重叠。严格企业类型支持的候选文件分别为汽车 259、半导体 120、能源 815 条。候选行业来源并不证明该企业只有这一行业、目前仍运营或所有关系在指定历史日期有效。

官方查询初始返回汽车 1,500（达到封顶，**不是全部汽车企业**）、半导体 231、能源 1,033 条候选。去重并加上真实产业/材料种子共请求 2,773 个种子 QID，然后仅请求 3,758 个一跳目标 QID（上限 5,000，不继续递归）。7 个 QID 在官方响应中 missing；没有补造。另为分类采集的类别实体严格封顶 2,500 个，不混入正式工业实体数。

## 可以直接使用的文件

- `normalized/nodes.jsonl`：正式实体，有 QID、名称、说明、类型证据、响应 SHA、抓取时间、领域候选来源。明确虚构对象已排除。
- `normalized/relations.jsonl`：两端均存在于正式 nodes 的唯一 `(subject, property, object)` 事实。保留原 Wikidata 属性语义及支持 statement ID；它是当前静态事实图。
- `normalized/relation_statements.jsonl`：上述事实的详细证据，逐条保留真实 `references`、`qualifiers`、rank、响应 URL / SHA / 时间；不同时间限定的 statements 可对应同一个事实三元组，分析时以此证据文件为准。
- `normalized/risk_labels.jsonl`：所有实体的标签都为 `label = null`、`label_code = -1`、`gold_label = false`。**未知不是负样本，不能当作训练或测试真值。**
- `normalized/domain_*_firm_candidates.jsonl`：类型证据支持的三领域企业候选；领域可能重叠。
- `normalized/summary.json`、`normalized/integrity_report.json`：实际数量、边语义、可用性限制及逐条与原始响应核对的结果。

原始当前快照及审计材料：

- `raw/`：官方 JSON 原始响应、部分原始 gzip 响应、每次响应的 provenance；成功数据保留 source URL、抓取 UTC、状态、字节数和 SHA。限流失败会记录、重试并保留当前采集版本的 attempts。早期采集脚本曾覆盖后续成功前的部分失败响应，因此不宣称完整网络请求史。
- 顶层 `entities.jsonl` / `relations.jsonl` / `statements.jsonl`：采集全集及完整 claims；供溯源，不保证所有边对象均已收齐，也不是正式可用闭合图。
- `normalized/all_nonfictional_*` / `unresolved_relation_statements.jsonl` / `unknown_targets.jsonl`：含未收齐目标的真实断言，**不在闭合训练图中新增占位实体**。
- `normalized/excluded_entities.jsonl` / `excluded_relation_statements.jsonl`：明确虚构对象及相连断言的审计记录，不算正式真实图。目前未发现明确虚构类型路径，文件为空；未分类对象的现实属性仍未被独立逐项核实。
- `checkpoint/`：首轮约 9 分钟时的可恢复部分快照；与最终正式文件数量不同，不能混用。

## 边语义和时间边界

`subsidiary`、`owned_by`、`parent_organization` 是公司结构事实；`country`、`headquarters`、`location` 是地域事实；`industry` 是产业断言；`product_or_material_produced` 保留 `P1056` 的产出语义，可能指产品、材料或服务；`manufacturer` 是制造商事实。这些都**没有改名成 Supplies、InputTo 或风险传播因果边**，不从持股、地理邻近或共同产业推导供应依赖。

抓取时间及 revision modified 是采集/修订时间，不是关系首次公开时间。关系有效期的 `P580/P582`、引用抓取日期 `P813` 等按原文保存在 qualifiers / references；不自动转为知识发布时间。所有正式记录 `knowledge_available_at = null`、`historical_prediction_eligible = false`。**不能把 2026 年现有图当作在 2018—2025 年预测时已知的图。**

本来源缺少论文需要的真实风险结局、五维韧性值、观测传播路径与时间安全的供应图。因此它支持真实事实图与实体采集研究，不支持有监督风险性能复现，也不能据此补造准确率、AUROC、F1 或原论文表 2 等实验分数。

## 重跑和独立核验

需要 Python 3.10+，只使用标准库，无账号、令牌或登录依赖。在本目录运行：

```powershell
python collect_wikidata.py --limit 1500 --related-limit 5000
python normalize_wikidata.py --taxonomy-limit 2500 --taxonomy-depth 8
python verify_real_graph.py
```

脚本优先重用带 provenance 的响应缓存。保留 `raw/` 时，本次工程可离线复查原始实体、实际关系、类型断言链、references / qualifiers、标签未知、闭合端点、数量和文件 SHA。`verify_real_graph.py` 失败退出码为 1。缓存删除后重新查询会得到新的当前快照，数量和内容可能改变；不是重建固定历史图。

采集器遵守每批最多 50 个实体、识别式 User-Agent、`maxlag=5`，遇到 HTTP 429/503 和 API maxlag/ratelimited 等待并重试；本次使用顺序请求。分类辅助采集和主采集最多两个请求进程，均不需要访问凭证。

## 来源和许可证

[Wikidata 官方 Data access](https://www.wikidata.org/wiki/Wikidata:Data_access) 明确其结构化数据使用 CC0；本包下载的实体标签、说明、别名、claims 及它们的原始响应属于 Wikidata 结构化数据，标记 `CC0-1.0`。没有下载被引用文章的正文、图像或新闻全文；引用 URL 不等于取得被引用内容的版权许可。

接口、限制和实现依据：[Wikimedia API rate limits](https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits)、[WDQS User Manual](https://www.mediawiki.org/wiki/Wikidata_query_service/User_Manual)、[maxlag 参数](https://www.mediawiki.org/wiki/Manual:Maxlag_parameter)、[Wikidata REST API](https://www.wikidata.org/wiki/Wikidata:REST_API)。实际选择器、搜索解析、请求 URL 和响应哈希分别在 `selectors.json`、`search_resolution.json`、`request_manifest.json` 及 raw provenance 中保存。

本目录新编写的采集、规范化和审计脚本按 `LICENSE_CODE` 的 MIT 许可证共享；源数据仍为 CC0，两者分别声明。Wikidata 是社区维护的真实采集事实来源，并非对每条现实世界断言作独立核实的金标准。
