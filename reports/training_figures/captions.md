# 完整模型训练历史图说明

`full_model_training_histories` 的八个面板来自 20 次完整模型正式训练日志：四个数据集各运行种子 17、29、43、71、101。每行左图为训练目标的批次平均损失，右图为验证集节点 Average Precision（AP；本工程表格记作 AUPRC）。颜色和线型共同标识种子，空心圆标记按验证 AP 与 `min_delta` 规则实际选中的 checkpoint。训练达到早停条件后，各曲线在各自真实记录的最后 epoch 终止。

所有曲线直接绘制原始日志，没有平滑、插值、补齐或外推。图中不显示均值、标准差、置信区间或显著性检验；不把不等长历史补齐后求平均。五个训练种子是模型优化重复，不能当作五个独立研究样本。训练损失包含节点、对齐及可用传播监督目标，不同数据集之间的损失数值不代表统一校准的误差尺度。验证 AP 用于反复选择 checkpoint，其峰值不应作为独立测试结果。

本图展示训练与验证诊断，不是受控参数敏感性、噪声鲁棒性或真实供应链因果机制证据。SIM 数据集为仿真；ICKG-Weak 为真实文本来源的弱标签任务，不能视为真实中断金标准。正式测试指标及其五种子样本标准差请查主实验表。

Source Data：`source_data/training_history.csv` 保留每次正式训练的全部逐 epoch 数值与 checkpoint 标识；`source_data/checkpoint_selection.csv` 保留选择结果、数据文件和输入日志校验和。`input_manifest.json` 记录 20 次 checkpoint 选择规则复核及全部源数据与导出图校验和。SVG 中的文字保持可编辑，PDF 嵌入 TrueType 字体，PNG 按 7.2 × 8.2 英寸、300 dpi 导出。
