# 交付文件验证与还原

本目录保存公开真实来源的采集代码、原始下载包、规范化数据及证据摘要。
规范化的 JSONL/CSV 以 gzip 保存，压缩包中不重复保留解压副本。
`public_file_manifest.json` 为每个文件记录保存字节数和 SHA-256；对于 gzip，
还记录原始解压字节数、SHA-256 和目标路径。gzip 使用固定 mtime=0，内容未改写。

## 先验证，再还原

在本目录执行：

```powershell
python verify_real_public_delivery.py
python materialize_real_public_data.py
python -m pip install -r requirements.txt
```

验证工具只读，不写入数据或报告。还原工具依据 manifest 解压 gzip，并从官方
NHTSA ZIP 还原原始 TXT；已有文件内容不同会拒绝覆盖。还原后会增加磁盘占用。
manifest 描述压缩交付态，解压得到的文件是衍生副本，不参与原始交付文件哈希清单。

## 采集与原始证据审计

汽车采集与证据核验脚本位于 `nhtsa_sources`，能源采集脚本位于本目录。
运行规范化程序或源记录核验前先完成上述还原步骤。
`energy_sources/evidence_verification.json` 是交付时冻结的独立能源源单元格审计结果。
`energy_sources/energy_qa.py` 保留原审计代码：它以脚本的上两级目录作为工程根，
读取 `energy_sources/raw` 及已还原的 `energy_sources/normalized`，并将新审计报告写入
脚本旁的 `energy_validation.json`。运行源审计会产生新报告；压缩交付验证仍为只读。
代码中保留的 `.runtime` 搜索路径是原采集环境的可选路径；本交付不包含该目录，
通过 `requirements.txt` 安装 `xlrd`、`openpyxl`、`requests` 即可提供这些依赖。

对官网索引 JS 仅提供公开 URL、下载时间、字节数及 SHA-256 元数据，未交付全文。
原始数据包及 API 响应保留官方记录；事实图关系含义和数据覆盖范围以各来源说明为准。
此工程不将召回、发电设备记录或公开知识库关系解释成真实供应链传播标签。
