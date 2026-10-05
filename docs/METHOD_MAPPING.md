# 论文与代码对应说明

本工程依据 49 服务器导出的 LaTeX 稿件 `20260708.tex`，题名 **Dual-Memory Resilience-Gated Overload Redistribution for Supply Chain Risk Prediction**。原项目 ID：`6a44bc3971659e98a23141e1`；该 TeX 的 SHA-256 为 `3546659d6662242201da1b9d79d91226ea787248a0005cac6254b5b78d87ce4b`。实现依据是正文第 3 节的公式及框架图，不依据稿件实验表反推或生成分数。

交付定位：DM-ROR 方法原型、可重复的数据构建与训练评估工程。论文原始 SC-Auto、SC-Semi、SC-Energy 数据、原作者运行日志及完整基线权重未找到；本工程不构成原论文表中结果的数值复现。模拟数据和真实文本弱标签的实验必须分别解释。

## 三阶段映射

| 稿件模块 | 实现位置 | 对应行为与工程选择 |
|---|---|---|
| Stage I：异构供应链长期记忆 | `dmror/model.py:HeterogeneousLayer`, `DMROR._long_memory` | 节点特征线性投影加节点类型 embedding；各关系独立消息矩阵，依赖权重归一化，按入边聚合，残差与 LayerNorm。稿件只写 HGNN，未指定 HGNN 架构；本实现选择 R-GCN 风格加权均值编码器，不等同于 HGT。 |
| Stage I：短期风险单元 | `dmror/build_simulated.py`, `dmror/build_ickg.py`, `dmror/extract_signals.py` | 模拟集来自明确标记的文本模板；真实公开文本按固定词汇规则产生风险片段。独立的本地 LLM 抽取工具支持事件、实体、类别、不确定性、情绪、置信度及逐字证据校验；该可选抽取不应被当作当前所有数据均由 LLM 抽取得到的证据。 |
| 时间窗与 Top-K 检索 | `dmror/build_ickg.py`，`memory_indices`, `delta`, `confidence`, `signal_mask` | 训练输入为预测时点之前的文本。真实文本按实体链接、严格前 30 天时间窗取最新 5 条，confidence 统一为 0.8，避免原图全期资料置信度污染；这是稿件 time-aware relevance 检索的简化。模拟集按生成流程放入当前警告与较早的例行更新。没有实现全图语义邻居扩展检索。 |
| 提示与 LLM 编码 | `dmror/encode_signals.py` | 本地冻结 LLM 对单条 evidence 文本编码；用 tokenizer 字符 offset 定位 evidence token，取最后层 hidden states 均值，经种子 731 的固定随机投影到 64 维并归一化。缓存可随工程交接；LLM 权重不包含在交付中。 |
| 完整实体—邻域—多信号提示 | `dmror/extract_signals.py` 的输入接口 | 工具接受实体描述、邻域摘要字段；当前已缓存实验只编码单条风险证据，未执行论文完整 `P(v,t)` 的实体与邻域联合编码。字段接口存在不能代表这项实验已完成。 |
| Stage II：时间注意力 | `DMROR._forward_single` | `q(hL)·k(z)/sqrt(d) - eta*age + confidence` 后按有效信号 softmax；空记忆返回零向量。学习 `signal_projection` 将离线编码映射到模型维度。 |
| Stage II：向量门控融合 | `vector_gate`, `short_projection`, `fusion_norm` | `g=sigmoid(W[hL;mS;b])`，`h=LN(g*hL+(1-g)*Ws*mS)`，与正文公式对应。 |
| Stage III：负载与吸收阈值 | `load_head`, `threshold_head` | `ell=softplus(Linear(h))`，`theta=softplus(Linear([b;h]))`，`overload=relu(ell-theta)`。实现采用带 bias 的线性头；稿件简写为无 bias 内积。初始化为正负载余量，避免初始 ReLU 全死造成传播头无法学习。 |
| 关系缓冲 | `buffer_head` | 输入源/目标表示、关系 embedding、依赖权重与 4 维关系缓冲，输出 sigmoid beta。 |
| 再分配系数 | `redistribution_head`, `group_softmax` | `chi` 具体选择源/目标短期记忆余弦相似度；按 **源节点和关系类型** 分组 softmax，再减 `lambda_beta*beta`。正文未指定 chi 的算法，本选择应作为实现细节报告。 |
| 单步负载接收 | `incoming.index_add_` | `incoming[v]=sum(overload[u]*Delta[u,r,v]*(1-beta[u,r,v]))`。HGNN 可多层，但负载重分配为正文规定的单次计算；路径检索串联边强度，不代表模型迭代执行了多轮物理级联。 |
| 节点预测 | `node_head` | 输入 `[h;ell;incoming;theta]`，输出风险 logit/probability。未来 30 天含义由数据的标签窗口决定，模型本身不接收未来证据。 |
| 路径强度 | `edge_score` | 保留正文 `pi=overload[src]*Delta*(1-beta)*node_prob[dst]`。框架图的简写公式省略目标概率因子，以正文公式为实现依据。 |
| 路径排序 | `dmror/paths.py:beam_search_paths` | 累加 `log(pi)` 等价于原始强度乘积；禁止重复节点，逐层 beam 截断，输出各长度候选的前 K 条。强度可以大于 1，不能解释为路径概率。 |
| 联合训练 | `dmror/losses.py`, `dmror/train.py` | 节点 BCE、余弦 InfoNCE、有效边 BCE 与可选 L2；实际训练用训练标签计算正类权重和 AdamW。损失用均值，稿件写求和；权重数值因此不能直接按论文求和式比较。 |

## 稿件需要说明的数学修复

正文给定的 `pi` 含 softplus 负载，**没有上界**。随后用 `log(1-pi)` 写 BCE 会在 `pi>1` 时失效。本工程保留 `pi` 为强度，用

```text
edge_score = pi
edge_prob = 1 - exp(-pi)
L_path = BCE(edge_prob, edge_label)
```

作为显式可运行修复。`edge_prob` 是单调有界映射，与 raw pi 排序一致；它未经过事件频率校准，不能直接当作现实传播概率。论文若使用本代码，应同步修订路径损失说明，不能称该修复为原稿原式。

关系级 softmax 分母分别属于每个 `(u,r)`，因此各关系分别可分出同一份 overload。总跨关系发送量不保证等于源 overload；本实现忠实保留该公式，不能宣传为全图质量守恒算法。阈值/缓冲头没有对韧性特征施加单调权重约束；真实训练后“更大韧性输入必然更小风险”的因果性质也未得到保证。

## 影响集合与路径的适用范围

正文的节点头和阈值集合 `S={v:p_v>=delta}` 是时刻 t 的**全图预测**，没有指定事件或源节点作为条件。`source_mask` 用于路径起点和评估，未进入 DM-ROR 节点前向计算。所以工程可以描述“未来高风险节点排序”和“从给定源出发的高分路径检索”，不能把全图排名直接表述为对某一源事件的反事实影响集合。多起点模拟样本尤其需要区分这两个含义。

`ranking_metrics` 的 P@K/R@K/NDCG@K 基于前 K 个已知标签节点。`threshold_scope_metrics` 单独计算正文阈值集合的 Jaccard，使用验证集 Macro-F1 选择的节点阈值；另保留 `jaccard_at_k` 作为 Top-K 集合指标，不混用两个定义。阈值 Jaccard 在真值和预测均为空时记 1，同时输出空集合天数及仅有正类天的 Jaccard。默认 K 为 `min(N,max(5,ceil(0.1*N)))`，报告应给出实际 K。

路径真值定义为：从 source 可达的已标注正边诱导图中，最多 `max_hops` 跳的全部简单路径与各正边前缀。默认最长 4 跳、beam width 10、每源最多 5 条。每条真值用节点序列表示，不区分同端点的不同关系；预测和真值均限制在已知边标签子图。Path@K 定义为有至少一条真路径的 source-time 中，top-K 候选命中至少一条精确节点序列的比例；无真路径的源不进入该分母。此定义是本工程可审核的评估约定，稿件未给出足以确认严格等价的 Path-F1/Path@K 定义。beam search 是近似候选搜索，不保证穷尽全图全长最优解；raw pi 的乘积也存在随长度与尺度变化的偏好。当前评估路径检索阈值为 0，所有已知标签正强度边均可成为候选，未强制按节点分类阈值切出高风险子图。

## 消融解释

| 模式 | 操作 | 解释边界 |
|---|---|---|
| `full` | 双记忆、向量门、阈值残差、关系吸收 | 当前实现的完整 DM-ROR 原型。 |
| `no_stm` | 短期记忆置零，使用长期表示 | 与稿件 w/o STM 思路对应；仍包含韧性和传播头。 |
| `no_ltm` | 长期表示置零，使用短期表示 | 仍使用边结构进行重分配，所以不是“完全不使用图”的纯文本分类器。 |
| `no_time` | 去掉注意力里的 eta*age | 已构建的 Top-K/时间窗保持不变；不是对所有时间机制的消融。 |
| `no_resgate` | 节点/关系韧性输入置零，theta=0、beta=0 | 去掉显式吸收，但依赖和图拓扑仍存在。 |
| `no_overload` | 发送量替换为 sigmoid(ell)，发送时 beta=0 | 作为工程定义的普通图扩散对照；不是 PageRank、Independent Cascade 或原稿某个已发布算法的同名实现。 |
| `avg_fusion`, `scalar_gate` | 均值、标量门 | 融合消融；共享其余 DM-ROR 结构。 |
| `no_align`, `no_path` | full 模型分别关闭 align 或 path loss | 网络前向不变；确认训练器没有把损失模式误传给模型枚举。 |

损失权重也支持设为零的诊断实验。是否已运行某个消融及其配置，以该次 `results` 输出中的 config、checkpoint 和 history 为准，不能从接口支持推断全部实验都已完成。

## 原稿基线覆盖

原稿列出的基线包括：

- 文本：TF-IDF+LR、BERT、RoBERTa、FinBERT、LLM-Classifier。
- 图：GCN、GraphSAGE、GAT、R-GCN、HGT、TGAT、TGN。
- 传播：PageRank、Random Walk with Restart、Independent Cascade、Linear Threshold、IIM、Stochastic Load Redistribution。
- 融合：Text+Graph Concat、BERT-GNN、LLM-KG、LLM-GNN。

本工程提供三个独立工程基线，位于 `train.py:IndependentBaseline`：

| 训练模式 | 实际算法 | 边/路径评分 |
|---|---|---|
| `text_mlp` | 有效 frozen LLM 信号均值后接 MLP，仅文本分类头 | 未训练的 `p(src)*p(dst)*clamp(dependency,0,1)` 启发式。 |
| `graph_only` | 静态关系 HGNN 后接 MLP，节点分数在各时刻相同 | 同上；图没有动态文本或 operational 特征。 |
| `concat` | 静态 HGNN 表示与文本均值拼接后接独立 MLP | 同上；没有 DM-ROR 韧性/负载/缓冲/传播头。 |

三个基线均不使用 align/path loss。其模拟路径分数只能作为明确注明的启发式对照，不等同于训练后的传播模型。DM-ROR 类内部保留的 `mode=concat` 是共享其余结构的融合原型；正常训练器的 `--mode concat` 路由到上述独立基线，不能混淆两者。

独立文本头不能命名 BERT、RoBERTa 或 FinBERT；DM-ROR 的 `no_stm` 是同架构消融，不能冒充独立发表的 R-GCN 基线。原稿所列 22 项独立基线均未按其完整发表算法重现，以上工程对照仅覆盖文本、图和融合三类思路；不声称复现“全部 SOTA 对比”。

原稿设想隐藏维度 128/256/512、学习率 1e-5 至 3e-4 及多组时间窗/检索 K 搜索；本工程采用较小的可交接实验配置，不能把未运行的网格搜索作为已完成的验证。真实参数与随机种子读取运行 config。
