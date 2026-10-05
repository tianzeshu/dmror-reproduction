本目录保存从 NHTSA 官方网络来源实际采集的汽车安全召回数据和 vPIC 制造商登记记录。没有模拟事件、生成摘要、伪造供应关系或负标签。

采集时间为2026年10月5日。召回子集按官方 `RCDATE`（Part 573报告接收日期）筛选2018年1月1日至2025年12月31日，车型年款不用于筛选时间范围。

完整召回批量源为 [FLAT_RCL_POST_2010.zip](https://static.nhtsa.gov/odi/ffdd/rcl/FLAT_RCL_POST_2010.zip)，来源及下载方式见 [NHTSA Datasets and APIs](https://www.nhtsa.gov/nhtsa-datasets-and-apis)，29字段定义保存在 `raw/RCL.txt`。ZIP原字节、解压原文本、下载URL、HTTP响应头、UTC时间和SHA-256均已保留。

实际采得185,699条官方原始记录行、8,090个不同召回campaign、899个字面报告制造商名称以及22,547个不同的“召回类型＋品牌＋型号＋年款”产品键。`9999`年款在官方字典中表示未知或不适用，归一化JSONL中为null，原始字段仍保留。

汽车召回数据包含车辆、设备、轮胎和儿童座椅。campaign数量分别为7,230、737、93、30。一个campaign可对应多个车型、年款、组件或记录行，因此185,699行不是185,699篇独立文本。独立事件以8,090个campaign计；每个事件保留官方原文summary、consequence、remedy、原记录列表和实际存在的文本/组件变体。

主要机器可读文件位于 `normalized/`：

- `recall_rows_2018_2025.jsonl`：逐行归一化记录；`source_fields`含全部29个原始字段，`evidence`含原文本字节偏移、行长、行SHA-256及文件SHA-256。
- `recall_rows_2018_2025.csv`：29个原始字段及行号、字节偏移和SHA-256。
- `recall_events_2018_2025.jsonl`：campaign去重的真实正事件和官方原文。
- `affected_products_2018_2025.jsonl`：来源明确的品牌、型号、年款、召回类别组合。
- `recall_fact_nodes.jsonl`、`recall_fact_edges.jsonl`：32,917个事实节点与88,959条来源支持边。
- `vpic_manufacturers_current.jsonl`：vPIC分页1–100的10,000个不同官方Mfr_ID，包含官方名称、国家和车辆类型。

召回事实图包括报告制造商、列明的被召回产品制造商、召回campaign、产品组合和官方组件描述。报告制造商字段MFGNAME与列明制造商字段MFGTXT作为不同来源角色保留，没有进行推断实体合并。因此899个报告名称与796个列明名称不能相加宣称1,695家独立企业。组件描述来自COMPNAME，不等于独立材料或部件供应商。

边仅表示官方记录支持的“提交召回报告”“召回列明产品”“涉及组件描述”“列明产品制造商”。它们不代表采购、供应依赖、供应链中断或真实风险传播路径。

vPIC制造商分页采集采用顺序低频请求，每个新请求开始至少间隔1.5秒；没有VIN查询。第100页仍有100条结果，故该登记集合明确是**当前2026年登记快照的部分分页覆盖**，不能宣称完整登记库或2018年的企业集合。其车辆类型混合乘用车、卡车、拖车、摩托车等，也包含没有已填车辆类型的登记记录。完整分页和逐页SHA-256见 `vpic_collection_manifest.json`。

标签只表示确实存在的官方召回正事件。未出现在召回集合中的企业/产品保持未知，不能自动当作真实负例。召回是产品安全监管事件，不等同于供应链中断。

当前下载是2026年数据快照，虽然报告接收日期在2018–2025年，但官方summary、remedy等叙述字段可能包含之后的修订。不能未经处理就把这些文本当作历史预测时刻已经可见的输入。

数据公开使用依据保存在 [Data.gov官方召回元数据](https://catalog.data.gov/dataset/nhtsas-office-of-defects-investigation-odi-recalls-nhtsa-api-6e97f) 的原始HTML中，其中`license`为`http://www.usa.gov/publicdomain/label/1.0/`。NHTSA [Terms of Use](https://www.nhtsa.gov/about-nhtsa/terms-use)也说明其公开信息可复制和分发。直接下载两份NHTSA网页HTML时返回403，失败记录保存在raw；批量数据、字典、Data.gov许可元数据和API使用政策已成功获取。

全量证据检查结果见 `evidence_verification.json`：185,699行逐行对照官方原字节和29字段；8,090事件逐项对照官方文本；88,959条事实边的所有支持记录已检查；10,000个vPIC制造商逐项对照保存的官方API页面。没有通过生成文字补齐缺失字段。

复核命令（Python 3.8及以上，仅标准库）：

```bash
python acquire_nhtsa.py
python normalize_nhtsa.py
python acquire_vpic.py --max-pages 100
python build_nhtsa_fact_graph.py
python verify_nhtsa_evidence.py
```

已有原始文件会按保存摘要校验并复用。官方数据持续更新，新的下载时间可能产生不同SHA-256和记录数量；本次快照以保存的原始文件和manifest为准。
