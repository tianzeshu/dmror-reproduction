# 公开版数据重建与冻结输入

直接训练和评估使用 `data/prepared/*/dataset.npz`；这四个文件是原实验使用的冻结 LLM 特征版本，字节及 SHA-256 保持不变。公开工程省略新闻原文及 `texts.jsonl`，因此从文本完整回放的前提与原内部交付不同，见 [PUBLIC_RELEASE.md](PUBLIC_RELEASE.md)。

## 已公开的冻结数据

| 数据集 | dataset.npz SHA-256 |
|---|---|
| SIM-Auto | 4e6d9baa6fb71bba375db77cf9c72ec9b862a37434715b36277dd7265588d017 |
| SIM-Semi | 54f9fd516dd6bda7be62f0c16561e9f6c0fe5841c8997a04b3cc0ffc1a5dc5da |
| SIM-Energy | 2ea23baaa883ac186aaa5b8c96004e77f63a0be974c8180f9bab964e9fcc4676 |
| ICKG-Weak | b6b22b31f55ed301f8fda3dfc3f26a6d69a1588d22a2781a6f8f3c2c5a631735 |

`embedding_cache.npz` SHA-256 为 `fcd062e8d781918da2ef5fe96f2015261a7c18925b4b11c45075b90705f062cf`。缓存是本地冻结 Llama-3-8B 的证据 token hidden-state 均值，经 seed=731 标签无关随机投影从 4,096 维映射到 64 维再归一化；最长 256 token。其 3,936 条原文未公开，不含端到端微调或完整邻域提示实验。

```bash
python -m dmror.data data/prepared/SIM-Auto/dataset.npz
python -m dmror.data data/prepared/SIM-Semi/dataset.npz
python -m dmror.data data/prepared/SIM-Energy/dataset.npz
python -m dmror.data data/prepared/ICKG-Weak/dataset.npz
```

## 合成场景生成

```bash
python -m dmror.build_simulated --output data/rebuilt/prepared
```

三套合成场景种子为 1201、1202、1203，各 112 节点、450 边、84 查询，独立随机传播标签及模板信号。基础构建输出 hash 诊断向量；它与正式 NPZ 的冻结 LLM 编码不同，不可混用已训练 checkpoint。要复跑已报告实验直接使用公开冻结 NPZ。

## 合法取得原文后重建 ICKG

原构建器逐项校验以下四个源文件；公开版不提供这四个文件。可向资料持有者在合法权限下取得同一历史快照，或者依法收集新来源并创建独立数据版本。来源核查元数据位于 `data/source_metadata/`。

| 文件 | 原文件 SHA-256 |
|---|---|
| canonical_entities_merged.jsonl | c246f1afbb8be6e6c956c1ed6b97467ba7d420b8ea022fc7af4da0483d5aa8a4 |
| kg_edges_high_precision.jsonl | 3ad550641dd590b490c47c194cd243db027be843aced1e25d358af1da8269cbf |
| contexts.jsonl | 4109e5df303e1edb9562d74fc518bdc6580463f3a0c5bf4735f97d25c3af0d91 |
| context_entity_links.jsonl | f64b5466ff4ee007028f2f72091d69f74e6725f4bbdb99f9859df0fbcfb29732 |

取得上述准确文件并放入本地 `data/raw/ickg/` 后可运行：

```bash
python -m dmror.build_ickg --raw data/raw/ickg --output data/rebuilt/prepared
```

同样先产生诊断特征。原 `dmror.apply_cache` 还需要另行合法取得完整 `texts.jsonl`（原 SHA-256 `90e3b400b0f81a6ab95e55cfb2731554250d9b1f6a52062a1b6973a45f09cf5d`），才可按完全相同的原文回填旧缓存。公开元数据不能替代原文匹配，当前网页或新抓取文本不保证与历史快照相同。原内部重建的逐数组/字节比对证据保留于 `REBUILD_VALIDATION.json`，这是历史记录，不表示公开版包含其全部前置文件。

合法添加或变更文本后，用独立输出目录、合法的本地模型和新版本编码：

```bash
python -m pip install -e ".[llm]"
python -m dmror.encode_signals --datasets data/rebuilt/prepared/ICKG-Weak --model /path/to/local/Llama-3-8b-instruct --output data/rebuilt/llm_cache --device cuda:0 --batch-size 8 --max-length 256
```

模型权重未再分发。改变输入、检索、时间窗或弱标签后应创建新协议并重新选参；没有实测韧性时保持缺失掩码，不以图度数或报道量伪装库存/缓冲观测。风险词规则、日期边界和标签窗见 `dmror/build_ickg.py`；容量、耦合和合成噪声见 `dmror/build_simulated.py`。
