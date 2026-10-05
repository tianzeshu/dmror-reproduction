# 正式训练源码快照

这14个模块逐字节取自正式实验执行后的49服务器，摘要与260次正式运行的记录一致。它们是历史追溯快照；日常安装、训练与重建仍使用工程根目录的 dmror 包。

当前训练核心与快照相同。之后仅 apply_cache 增加跨平台 NumPy ZIP 元数据修正，使Windows重建数据也获得相同文件SHA；此工具不参与训练计算。另新增人工标签导入与绘图/验收工具。详见 docs/POST_TRAINING_CHANGES.json 和 docs/HANDOFF_CHECKS.md。
