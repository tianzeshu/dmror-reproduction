# 人工审核的真实结局标签接入

本工程现有 ICKG 标签是未来文本中的字面风险词出现规则，不能解释为实际供应中断。`dmror.import_annotations` 允许把经过人工审核的实际结局导入**新的**数据集；原始数据集、文本标签规则和未来文本证据文件保留。这里只提供空白模板及软件验证，没有完成任何真实事件标注，不能把模板或测试数据作为真实实验结果。

## 节点标签

复制 `data/annotation_templates/node_labels.csv`，根据 `nodes.jsonl` 的 `entity_id` 标注。NPZ 若提供 `node_ids`，以该顺序为准。`query_day`、`target_day` 必须精确对应 NPZ 的 `prediction_days`、`target_days`；可以填写 `YYYY-MM-DD` 或 1970-01-01 起的整数天数。ICKG 的原目标窗为未来 30 天 `[query_day,target_day)`，包含起点、不含终点；导入器不改变任何目标窗。

| 字段 | 内容与理由 |
| --- | --- |
| `entity_id` | 数据集已有实体 ID；名称不能替代 ID。 |
| `query_day`, `target_day` | 本条标签明确覆盖的完整预测窗；不能近似匹配或自动移到邻近窗口。 |
| `label` | 只能为 1 或 0。1 表示该实体在目标窗中发生已审核的风险事件；0 表示审核者有足以覆盖完整目标窗的证据，并明确确认没有符合定义的事件。 |
| `event_date` | 正例必填，必须落在目标窗内；负例留空。发布日期不能自动替代事件发生日期。 |
| `evidence_url` | 可追溯的证据地址。负例应指向全窗口覆盖材料或审核记录。 |
| `evidence_span` | 支持实体、事件日期、结局判断的原文摘录或明确定位；负例写明全窗口覆盖范围及审核依据。含逗号或换行时按 CSV 规范加双引号。 |
| `reviewer` | 实际审核者的姓名或稳定编号，不能为空。 |
| `verified` | 只能为 1，表示审核者签署上述判断。未审核草稿不能导入。**label=0 行的 verified=1 是对整个窗口覆盖的明确签署**，不是“未来没有检索到文字”。 |

没有人工标注的单元格是 UNKNOWN（-1），包括尚未审阅的负例。导入器不会继承原来基于文本缺失的 0/1 标签，不会将未观察到事件转为负例。每个实体、目标窗口只允许一条已合并证据记录；重复或相互冲突记录均拒绝。

审核者应先固定“风险事件”的操作定义，例如供应商确实停止交付而非媒体讨论或假设。需要独立核对归属实体、事件发生日期、来源可靠性以及覆盖范围。软件只校验格式和时间边界，**不判断原文是否真实、不认证数据成为金标准**。不得为了改善指标选择或改写标签；如调整定义，使用新标注版本并保留依据。

## 可选关系标签

`edge_labels.csv` 为每条已审核的有向关系提供精确窗口标签：`query_day,target_day,source_id,target_id,relation_type,label,evidence_url,evidence_span,reviewer,verified`。

`source_id → target_id` 与 `relation_type` 三元组必须精确且唯一匹配 NPZ 图及 metadata 中的关系类型。相同两端点的 Supplies、InvestsIn 等平行关系分别匹配；如果同一三元组有多条边，程序拒绝歧义。关系正例需要审核风险沿该关系传播的证据；节点同时为正并不足以证明传播。负例同样需要完整目标窗的明确审核，遗漏均为 -1。

这些是部分关系结局标签，不会自动构造整条因果路径或完整受影响范围。`source_mask` 从原数据集原样复制；ICKG 中它全部为 false，不能从未来标签反推源头。韧性、依赖权重和关系属性的 observed masks 也原样复制，未观察到的属性仍然未观察到。

## 导入与复现

在工程根目录执行，替换实际标注文件路径：

```bash
python -m dmror.import_annotations \
  --dataset data/prepared/ICKG-Weak \
  --node-labels my_annotations/node_labels.csv \
  --edge-labels my_annotations/edge_labels.csv \
  --output data/prepared/ICKG-Reviewed-v1
```

没有关系标注时省略 `--edge-labels`。输出目录必须不存在，且不能位于输入数据集内部；原始输入文件哈希与 metadata 不一致时拒绝导入。仅有空模板时也拒绝生成“已审核”数据集。

新目录包含可直接供现有训练器读取的 `dataset.npz`、更新的 `metadata.json` 与 `audit.json`、逐条 `annotation_evidence.jsonl`、原样保存的 CSV 和已有图/文本辅助文件。除了 `labels`、`edge_labels`，NPZ 中所有数组保留：特征、信号、memory indices、时间戳、划分、source masks、observed masks 均不从未来标签更新。`text_encoder` 与原始来源信息保留，metadata 记录输入 NPZ/metadata 和标注 CSV 的 SHA-256，输出 NPZ 另有新 SHA-256。

原有 `label_evidence.jsonl` 仅是**原文本代理标签的来源记录**，不能作为新增人工结局标签的验证依据；接受的新标签证据仅在 `annotation_evidence.jsonl`。原始未来文本规则不被重写。

训练前检查每个 train/validation/test 划分中的正、负、未知数量；某个划分仅有未知，或缺失任一类别，可能无法计算 AUPRC、AUROC 或正常选模型。应收集独立、覆盖适当的人工标签，而不能填充未审阅负例。未知标签应由现有损失和评价逻辑排除。局部人工标签仍不能消除 ICKG 的回溯选图、实体归一化及历史采集时间不确定性；数据类别也不会被软件自动升级为真实因果金标准。

运行 `python -m pytest tests/test_annotations.py -q` 验证结构行为。测试中的虚构实体和证据地址只用于程序测试，不是人工标注数据。
