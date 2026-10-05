# 原始构建的来源审计记录

本文件记录 2026-10-02 的原内部构建与定向交接，不是公开副本的内容清单。文中原始正文、LLM 抽取样例和服务器资料未公开；公开内容及重建前提以 [PUBLIC_RELEASE.md](PUBLIC_RELEASE.md) 为准。路径和摘要仅作历史来源核查。

审计日期：2026-10-02（Asia/Shanghai）。服务器通过既有 SSH alias `draft-remote` 只读访问；未读取凭据或 token，未修改服务器文件。仅将下列科研数据复制到当前交付目录用于已授权的学生工程交接。

## 结论

可以复用带时间、来源 URL、实体链接和文本证据的 ICKG 工业资料，但未找到论文所称的三个完整 SC 风险金标准数据集。现存 ICKG 关系为 LLM 抽取候选及规则筛选，未见完成的人工正确性标注；不存在经验证的节点中断、真实缓冲能力、影响范围或因果传播路径标签。因此，该资料只能支持明确命名的“未来风险相关报道”弱标签预测，不能把该标签解释为实际供应链中断。另行构建的模拟数据须明确标为 synthetic。

## 可复用资料

| 服务器文件 | 数量 | 关键字段与用途 |
|---|---:|---|
| `/root/ICKG/phase02_filter_compute_minerals/output/documents.jsonl` | 8,171 文档，其中 8,165 有日期 | `doc_id,date,publish_date,title,text,url,source_type,matched_groups,matched_keywords`；真实采集文本，日期 2015-01-01 至 2025-09-02。news 5,255、policy_candidate 2,829、announcement 55、patent 32。policy_candidate 是筛选类别，不是政府政策真实性认证。 |
| `/root/ICKG/phase03c_entity_canonical_consolidation/output/canonical_entities_merged.jsonl` | 8,852 实体 | `entity_id,canonical_name,entity_type,aliases,country,sector,source`；Company 8,769，其余 83 是技术、行业、材料、产品等。别名多为 LLM reviewed，不等于人工审核。 |
| `/root/ICKG/phase03c_entity_canonical_consolidation/output/entity_mentions_merged.jsonl` | 134,667 mention | `doc_id,entity_id,mention_text,start_char,end_char,date,confidence,grounding_method`；可按文本跨度追溯实体。 |
| `/root/ICKG/phase05_build_icr_snapshot/output/kg_edges_candidate.jsonl` | 4,251 边，2,061 图节点，7 关系 | `head_entity_id,tail_entity_id,relation_type,first_event_date,edge_time`。关系：Cooperates 1,320，Supplies 978，Competes 769，InvestsIn 556，Holding 394，Acquires 130，JointVenture 104。 |
| `/root/ICKG/phase05_build_icr_snapshot/output/kg_edges_high_precision.jsonl` | 2,236 边 | 规则高精度子集，Supplies 502。high_precision 是自动筛选名，并非金标准。日期 2015-01-19 至 2025-08-30。 |
| `/root/ICKG/phase05_build_icr_snapshot/output/edge_evidence_leakage_safe.jsonl` | 4,386 证据 | `edge_id,doc_id,event_date,evidence_span,confidence`；全部 `is_after_edge_time=false`，但仍须按预测时点筛选。 |
| `/root/ICKG/phase06_context_repository/output/contexts.jsonl` | 18,486 片段 | `context_id,context_date,doc_id,text,title,url,source_type,anchored_entity_ids`。entity_sentence 10,228，evidence_window 3,292，evidence_sentence 3,153，doc_lead 1,813。日期 2015-01-19 至 2025-08-30。 |
| `/root/ICKG/phase06_context_repository/output/context_entity_links.jsonl` | 28,360 链接 | `context_id,entity_id,context_type,context_date,doc_id`；适合将时间文本附到图节点。 |

原图按 unique dates 70/15/15 分割：train 2,170 边（2015-01-19 至 2024-09-27），valid 553（2024-09-28 至 2025-04-19），test 1,528（2025-04-20 至 2025-08-30）。本次未来报道任务需重新定义预测时间与标签窗口，不能直接套用原关系补全划分。

## 补充新闻资料

`/home/yinw/ICKG/06_ickg_icr_news_augmented` 包含公开网页增补流程：

- `phase01_normalize_public_news/output/normalized_public_news_documents.jsonl`：10,011 文档，2012-01-02 至 2026-06-19，含 `publish_date,date,title,text,url,canonical_url,language,source_domain,crawl_time`。原始抓取时间 2026-06-22。网页发布日期与可获得日期应分开解释，不能声称当年已经实时采集。
- `phase03_entity_link_augmented_news/output/entity_linked_documents.jsonl`：4,181 已链接文档，8,186 unique doc-company links。
- `phase03_entity_link_augmented_news/output/relation_ready_documents.jsonl`：1,968 至少含两个可用公司实体的全文，2014-01-28 至 2024-09-27；字段 `doc_id,date,title,url,companies,text`。按原 cutoff 2024-09-27 过滤。
- `phase04_relation_extract_augmented_news` 仅见 relation-ready samples 和输入报告，未见完整新关系抽取结果；不能把待执行流程算成已完成 KG。

source_manifest 引用 Windows `D:/project/ICKG_sync/...` 路径；这些是先前构建机的来源路径，不能假定对应原始政策全文已在 49 服务器存在。

## 标签真实性与时间风险

`phase04_relation_extraction/output/manual_eval_relations.jsonl` 有 300 条待审样本；四个 `manual_*` 字段填入数均为 0。现存关系不具备完成人工金标准的证据。风险、实际库存/替代性/资金/地域缓冲能力、受影响集合和因果路径均无完整观测标签。

原 edge 的 `evidence_count,doc_count,last_event_date,max_confidence,avg_confidence,weighted_support_score` 聚合了完整时段；snapshot_report 明确列出 1,141 条 future evidence。不能将这些全时段聚合属性输入早期预测。图必须冻结到最早 query 之前，文本必须满足 `context_date < query_time`。`high_precision` 子集的筛选本身用了完整证据，是回顾式资料筛选，不能描述为真实在线获得的图；报告应明确这一局限。

预 2019 高精度图仅 67 节点/70 边；预 2022 为 191 节点/197 边；预 2023 为 324 节点/345 边。工程采用最早 query=2022-01-01，冻结预 2022 边，并仅依训练期可用性选节点。train=2022，valid=2023，test=2024 至 2025-08-30；30 天标签窗口跨年边界时 purge。若预测每 10 天取一次，窗口重叠产生相关样本；随机种子方差不是独立数据样本置信区间。

本次弱标签定义应为：节点在未来 30 天是否出现预先固定词汇规则判为 risk-related 的公共报道；保存匹配短语、原始跨度、URL、日期与规则版本。未报道不等于未发生风险。韧性观测值缺失时填 0 并标记 observed mask=false；禁止将图度数包装成实际库存/缓冲能力。边/路径未知标签用 -1，不报告因果路径精度。

## 本地复制与 SHA-256

复制目录：`delivery/data/raw/ickg/`。复制后必须校验下列 digest：

| 文件 | SHA-256 |
|---|---|
| `kg_edges_high_precision.jsonl` | `3ad550641dd590b490c47c194cd243db027be843aced1e25d358af1da8269cbf` |
| `contexts.jsonl` | `4109e5df303e1edb9562d74fc518bdc6580463f3a0c5bf4735f97d25c3af0d91` |
| `context_entity_links.jsonl` | `f64b5466ff4ee007028f2f72091d69f74e6725f4bbdb99f9859df0fbcfb29732` |
| `canonical_entities_merged.jsonl` | `c246f1afbb8be6e6c956c1ed6b97467ba7d420b8ea022fc7af4da0483d5aa8a4` |

网页资料原始再分发许可未单独核实；本次为用户已有研究资料的定向工程交接，未外部上传。公开发布时须单独处理文本再分发许可；不能称原新闻正文为本工程自有开放许可。

## 模型与 GPU 环境

以下仅验证本地模型文件与环境存在，未下载权重、未安装依赖：

- `/mnt/ssd/model/Llama-3-8b-instruct`：4 个 safetensors，16,060,556,376 bytes，非量化，适合直接加载冻结 hidden states，tokenizer 与 config 同目录。不得随本工程复制权重。
- `/mnt/ssd/model/Llama-3.1-8B-unsloth-bnb-4bit`：1 个 safetensors，5,964,186,429 bytes；需 bitsandbytes，但本次检查的 5090 环境未装该依赖。
- `/mnt/ssd/model/Qwen3-32B`：65.52 GB 权重；`Qwen3-32B-unsloth-bnb-4bit` 实际 39.31 GB；`Qwen3-VL-30B-A3B-Instruct`、FP8、Qwen3.5-27B、Qwen3-Next-80B-A3B 亦存在。未发现更小 HF 文本模型。
- `/root/model/Qwen3-32B-unsloth-bnb-4bit` 另有原位置，SSD 位置适合读取。
- `/root/miniconda3/envs/vllm5090cu129/bin/python`：torch `2.11.0+cu129`、numpy `2.2.6`、transformers `4.57.6`；compiled arches 包含 sm120/compute120。实际 RTX 5090 上 8x8 CUDA matmul 输出 sum=512，通过。
- `/root/miniconda3/envs/vllm5090/bin/python`：torch `2.12.0.dev20260407+cu128`、numpy `2.4.4`、transformers `4.57.6`，compiled sm120，实际 CUDA matmul 同样通过。
- 两个 5090 环境都没有 scikit-learn、accelerate、bitsandbytes、sentence-transformers。`scope` 无 torch/numpy。适合使用 NumPy 与直接 `.to('cuda')`，避免假设额外依赖。

`/root/projects/APF` 只发现两个 dataloader 和 Reddit 数据文件，未找到供应链风险实现。目录名 risk/dmror/dm_ror/yuntong 的有限深度检索未找到对应旧模型工程。该结论只覆盖本次实际检查目录，不保证服务器任意位置不存在旧副本。

## 本次实际构建与核验

已执行 `python -m dmror.build_ickg`，输出 `delivery/data/prepared/ICKG-Weak/`。原始四文件的 SHA-256 与上表逐项一致。最终冻结图为 191 个 Company 节点、197 条关系边；全部节点已有 2022-01-01 之前的带日期链接片段，节点入选未使用验证、测试期活动或正标签。实体类型仅为企业，本资料不构成论文所有异构节点类型的实证验证。

预测时间为 2022-01-01 至 2025-07-24，每 10 天一次，共 131 次；30 天未来窗口。训练保留 34 次、验证保留 33 次、测试保留 58 次，另有 6 次因标签窗触及或跨越年界而清除。分别有 40/6,494、64/6,303、228/11,078 个正节点标签，类别极不平衡。这个分母是节点-查询对数，不能理解为独立实验样本。清除的查询另有 14 个正标签，未用于训练或评估。

记忆严格取公开标记日期位于 `(t-30,t)` 的片段，按日期取最新 5 条；共 5,099 个被检索的实体-片段记录、11,624 次检索观测。每条证据置信度固定为 0.8，仅作占位权重，不代表人工确认可信度，不读取来源中可能依赖完整证据的 `quality_weight` 或关系聚合置信度。`signals.jsonl` 保留实体、片段、文档、URL、日期；基础构建产生 64 维哈希诊断输入，正式冻结 LLM 缓存由单独的 `dmror.encode_signals` 执行并记录模型与权重来源。

固定 30 个中英文短语组成 `ickg-risk-publication-literal-v1` 规则，未来区间 `[t,t+30)` 的实体链接片段命中即为 1；0 仅表示收集语料中未记录到匹配片段。936 条标签支持证据保存于 `label_evidence.jsonl`，包括匹配短语、片段及文档跨度、日期和 URL。该规则不判断否定、假设句或多企业上下文中的实际事件归属，不能作为真实企业中断的人工金标准。标签生成没有使用 LLM 抽取后的当前风险字段作为未来标签。

静态输入仅为节点类型 one-hot 加零占位；未使用企业名嵌入、最终 mention/doc 数、国家/行业字段、图度数等代理缓冲值。关系依赖强度为常数 1，四维关系属性为 0，五维韧性为 0，各自 observed mask 全为 false。所有 `edge_labels=-1`，`source_mask=false`；不提供真实路径准确率。弱资料的 scope 只能解释为“将来出现词汇规则报道的节点集合”。

泛用数据契约核验及额外逐条源日期核验均通过：每个记忆 timestamp 严格早于查询且在回看窗内，所有正标签均有未来匹配证据，图边首次日期早于最早查询，跨分割目标窗均被 purge。基础 `dataset.npz` SHA-256 为 `459d505f2101eff033920daf418b1f261a25acf83bc8fe0af97902fb575e3e10`；后续 LLM 缓存替换信号向量后摘要会变化。以上检查**不能消除 high_precision 与实体合并在完整语料上回顾式构建的选择偏差**，metadata 明确标为未消除。

另生成 `delivery/data/llm_extraction_example/input.jsonl`：30 条 2022 年片段，其中 14 条命中固定词表、16 条未命中。输入保留 `document_id,text,timestamp,entities[{id,name}]`，不将选择组别传入 LLM。该有界样例用于验证真实本地冻结 LLM 的结构化抽取流程，不宣称有人工抽取金标准，不用于未来风险目标生成。选择日期包括 2022 年末被 purge 的片段，样例与训练目标独立；其 manifest 记录这一采样规则与 SHA。
