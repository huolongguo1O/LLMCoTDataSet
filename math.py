from datasets import load_dataset

# 加载数据集（自动识别train split）
dataset = load_dataset("ServiceNow-AI/R1-Distill-SFT", 'v0', split="train")

# 打乱数据顺序（不设种子以获得随机性）
shuffled = dataset.shuffle()
# 字段重命名与列过滤
processed = shuffled.map(
    lambda item: {
        "text": f"User: {item['problem']}\n\nAssistant: {item['reannotated_assistant_content']}"
    },
    remove_columns=shuffled.column_names  # 自动移除原始字段
)

# 保存为JSON Lines（自动处理大文件分块）
processed.to_json(
    "3.jsonl",
    num_proc=8,          # 推荐设置为CPU物理核心数
    force_ascii=False    # 保留数学符号和代码字符
)
