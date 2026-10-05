# Git 精简目录与完整 Release

本目录仅包含真实来源的采集/处理代码、实际统计、来源和证据核验元数据。
全部归一化 JSONL/CSV gzip 与原始官方 ZIP、XLS、API 响应在
[v1.2.0 Release](https://github.com/tianzeshu/dmror-reproduction/releases/tag/v1.2.0) 中。
只需归一化数据可下载 Release 的小型数据包；逐来源复核或完整工程运行使用完整工程分卷。

`GIT_FILE_MANIFEST.json` 仅列本 Git 目录实际存在的 identity 文件，不把 Release 数据列成 Git 文件。
在仓库根运行 `python scripts/verify_real_git.py`，可核验本目录代码与元数据的大小和 SHA-256，
无需下载或展开 gzip。`README.md`、各来源 README 和来源 manifest 保留完整冻结工程的说明与原始摘要；
它们提及的原始/归一化文件不在 Git 子集中。完整还原、原始字节/单元格/断言核验命令在 Release 完整工程执行。

历史模拟数据仍在仓库其他目录，未混入本目录；没有真实风险训练结果或生成负标签。
