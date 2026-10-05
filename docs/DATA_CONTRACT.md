# 数据契约与来源

数据文件为无 pickle 的 `dataset.npz`；加载须用 `np.load(..., allow_pickle=False)`。文本、来源与观察记录保存在 JSONL，统计、标签定义、哈希和局限写入同目录 `metadata.json`。训练结果必须绑定数据 SHA-256，避免文本重编码后误把不同数据版本当成同一实验。

## 统一数组

符号：T 为预测快照数，N 为节点数，E 为有向边数，K 为每节点最多检索文本数，D 为文本特征维度，F 为静态节点特征维度。以下为当前训练器读取的字段。

| 字段 | 形状 | 类型与含义 |
|---|---|---|
| `node_features` | `[N,F]` | float32；预测时可用的静态节点描述，不能来自未来标签或全时段统计。 |
| `node_type` | `[N]` | int64；从 0 开始的节点类型 ID。 |
| `src`, `dst`, `edge_type` | `[E]` | int64；有向边端点与关系 ID，节点索引必须位于 `[0,N)`。 |
| `edge_dependency` | `[E]` | float32；依赖权重，必须说明真实测量或代理定义。 |
| `edge_features` | `[E,4]` | float32；关系缓冲输入，未知值填零不等于实测缓冲为零。 |
| `resilience` | `[T,N,5]` | float32；库存、替代、产能、资金、地域五类输入槽。模拟集为观测模拟容量；真实公开文本集缺少实测值，填零并在伴随 mask 标记缺失。 |
| `signals` | `[T,N,K,D]` | float32；历史风险证据向量，padding 为零；编码器类型须从 metadata 读取。 |
| `signal_mask` | `[T,N,K]` | bool；只有 true 槽位才是有效文本。 |
| `delta` | `[T,N,K]` | float32；预测日减文本日，有效槽必须严格大于 0，单位天。 |
| `confidence` | `[T,N,K]` | float32；提取/规则置信度代理，不能等同于人工标注的概率校准值。 |
| `memory_indices` | `[T,N,K]` | int64；对应 `signals.jsonl` 的 signal_id；padding 为 -1，用于重编码与证据回溯。 |
| `source_mask` | `[T,N]` | bool；模拟风险起点或弱标签任务的历史观察起点；供路径评估用，不作为节点模型的额外输入。 |
| `labels` | `[T,N]` | float32；1=该数据定义的未来正类，0=该数据定义的负类，-1=未知。必须按数据来源解释 0/1。 |
| `edge_labels` | `[T,E]` | float32；1=已定义的传播正边，0=已观测模拟负边，-1=未知。真实公开文本没有传播标注时全部 -1。 |
| `split` | `[T]` | int64；0=train、1=validation、2=test、-1=边界 purge 快照。 |
| `prediction_days`, `target_days` | `[T]` | int64；预测日与标签窗口终止日，终止日不能跨入后一数据划分的预测时点。具体日历原点读 metadata。 |

附加字段可包括 `resilience_observed_mask`、关系缓冲观测 mask、实体 ID 映射和路径起点 mask。当前模型没有专门学习 missingness mask；真实韧性缺失应在实验解释中披露。

## 三个独立模拟数据集

`SIM-Auto`、`SIM-Semi`、`SIM-Energy` 均由 `dmror.build_simulated` 构建，是合成供应链，名称仅表明三个模拟场景。**它们不是论文的 SC-Auto、SC-Semi、SC-Energy。**

默认每集 112 节点、450 有向边、84 快照、30 天预测窗口、10 天查询步长；训练/验证/测试有效快照为 47/14/17，另有 6 个边界 purge 快照。5 类节点为企业、产品、材料、行业、地域。种子分别为 1201/1202/1203；参数差异是容量水平、依赖耦合、噪声与提示模板。

未来标签来自独立离散模拟器：初始扰动、延迟 Bernoulli 传递、库存递减及失败阈值。生成器没有调用 DM-ROR 的神经网络、权重或 softplus/门控公式。历史警告是潜在扰动的有噪声观察，警告时间早于 query；未来标签窗口内所有节点的模拟失败状态均已知，因此 0 有完整模拟观察含义。边正例指成功传入刚失败目标的模拟传递边；多个传入负载同时耗尽库存时可以有多条贡献边，不能解释为唯一原因归因。

模拟文本来自少量模板，实测容量由独立抽样产生。此数据适合检查模型可训练、传播模块数值行为及消融差异。其结果不证明模型在真实企业、新闻或供应链中具有同等精度，也不能证明在完整真实标注集上超过原稿基线。

## ICKG 真实文本弱标签

原始来源为用户 49 服务器已有 ICKG 资料。原内部构建复用的四文件为文本 contexts、实体链接、canonical entities 和自动筛选 high_precision 边；公开版省略 `data/raw/ickg/`，直接使用冻结 NPZ，来源元数据位于 `data/source_metadata/`。`high_precision` 是自动筛选的文件名，不代表人工金标准。

源资料审计得到 18,486 文本片段、28,360 实体—文本链接、8,852 规范实体和 2,236 自动筛选边。图关系与实体链接来自前序机器抽取及规则筛选；300 条待人工评估关系的人工字段未填写。既有资料未提供实际中断、真实库存/资金容量、事件受影响集合或因果传播路径标签。

构建产物名称为 `ICKG-Weak`。191 个 Company 节点、197 条 2022 年之前的自动筛选边；实体在最早 query 2022-01-01 之前均有带日期文本。131 个每 10 天查询快照，其中 train 34、validation 33、test 58、purged 6；有效训练期 2022 年、验证期 2023 年、测试期 2024 年至 2025 年 7 月。各期节点正例 40/64/228。原始全量材料的数量不能当作此最终评估图规模。

本工程弱标签的目标是 **未来 30 天是否出现规则判为风险相关的公开报道**。正类说明公开文本触发固定风险短语规则；负类说明该观察语料中没有这样的未来报道，不能解释为企业“没有真实风险”。该任务容易受到媒体覆盖和规则定义影响。规则用于构建 future labels；输入文本只用 query 之前的记录，保存日期、URL、原文和匹配短语，可审核其对应关系。

规则是逐字匹配，版本 `ickg-risk-publication-literal-v1`；没有处理否定、假设性讨论或事件到底属于段落中哪一家企业。实体链接到带风险词片段即构成该实体的报道代理正例，所以必须审核标注精度，不能以机器判词替代实际中断或事件归属标注。

实际静态节点特征为 8 维类型 one-hot 与零槽位；dependency 统一为 1，dependency observed mask 为 false；关系属性全部为零且 observed mask=false。resilience 全部零，`resilience_observed_mask[T,N,5]=false`。confidence 统一为 0.8，实体时间窗为 `(query-30天, query)`，检索最新 K=5 条。此简化没有使用原边全期资料的质量聚合，仍不能消除原始 high_precision 筛选的回顾式选择偏差。

真实图必须冻结到最早预测时点之前，并禁止把全时段 `last_event_date`、`evidence_count`、平均置信度等统计加入早期预测。高精度子集的原始筛选仍使用了回顾式全期资料，故本工程是历史公开资料回顾实验，不能声称图在当时已以同样方式在线获得。网页发布日期亦不等于抓取/可获得日期。

缺失 operational resilience 时使用零占位及明确的 observed mask；图度数、文档数不可改名为库存、供应商替代性或资金稳健性。真实集的 edge labels 为 -1，source_mask 全 false，并跳过传播路径质量指标；从学习强度另行检索出的路径可以作为未验证候选，不能包装为真实因果真值。

原新闻文字的再分发许可未逐项核实；当前交付是用户已有研究资料的定向科研交接，未将新闻正文视为本工程自有开放许可。若公开发布工程，须先根据各来源许可处理正文，可以保留来源 URL、元数据及重建脚本。

## 时间与训练约束

1. 训练、验证、测试按时间排序；每一快照的有效文本日期严格 `< prediction_day`。
2. 标签窗口跨划分边界的快照标为 -1 并从训练/验证/测试中剔除。需满足 `max(train_target_days)<min(validation_prediction_days)`，验证—测试同理。
3. 文本 scaler 只使用训练快照中有效文本拟合。静态节点 scaler 在预测时已知的全体图节点上拟合，属于 transductive 设置，不能宣称对未知节点的 inductive 泛化。
4. checkpoint 只按验证 AUPRC 选择；节点/边阈值只在验证标签上调；测试指标不参与训练或选模。
5. 五次不同模型随机种子衡量优化变动。固定同一模拟图种子与重叠窗口时，模型种子标准差不是总体置信区间；独立图重复、时间 block bootstrap 和更完整真实标注属于后续验证。
6. `signal_mask=false`、`label=-1` 与真实为零是不同状态。节点未知标签不参加分类指标；全部未知边应跳过路径监督与路径评估。

`dmror.data.audit_dataset` 检查维度、索引、标签域、时间边界和文件哈希；训练器也会检查关键数组及 split。通过这些检查只证明数据满足运行契约，不能证明机器抽取、语义标签和现实因果关系正确。

## 编码版本与来源验证

诊断构建可使用 SHA-256 signed hash 文本向量，metadata 明确标为 diagnostic。冻结本地 LLM 编码执行后，metadata 应更新模型名称、配置/权重 SHA-256、hidden size、池化说明、投影 seed、cache SHA-256 和 token 长度。禁止把初始 hash 版本的结果当成 LLM 实验。

源文件 SHA-256：

| 原始文件 | SHA-256 |
|---|---|
| `kg_edges_high_precision.jsonl` | `3ad550641dd590b490c47c194cd243db027be843aced1e25d358af1da8269cbf` |
| `contexts.jsonl` | `4109e5df303e1edb9562d74fc518bdc6580463f3a0c5bf4735f97d25c3af0d91` |
| `context_entity_links.jsonl` | `f64b5466ff4ee007028f2f72091d69f74e6725f4bbdb99f9859df0fbcfb29732` |
| `canonical_entities_merged.jsonl` | `c246f1afbb8be6e6c956c1ed6b97467ba7d420b8ea022fc7af4da0483d5aa8a4` |

以交付中的 provenance 和结果 config 为最终数据版本记录。LLM 重编码、Top-K、规则或划分变化会改变 dataset SHA-256，应创建新的结果目录并重新执行比较。
