from transformers import BertTokenizer, BertForMaskedLM
import torch

# 加载 BERT tokenizer 和预训练模型
tokenizer = BertTokenizer.from_pretrained("./pretrained_model/bert-base-chinese")
model = BertForMaskedLM.from_pretrained("./pretrained_model/bert-base-chinese")

# 示例中文新闻句子，含有两个 mask
news_sentence = "中国的[MASK][MASK]实力在全球范围内不断增强，成为了世界经济的重要支柱之一。"

# 将句子转换为 token，并找到所有的 [MASK] 的位置
tokenized_text = tokenizer.tokenize(news_sentence)
mask_indices = [i for i, token in enumerate(tokenized_text) if token == '[MASK]']

# 将 token 转换为索引
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])

# 预测每个 mask 处的词语
for mask_index in mask_indices:
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0][0, mask_index].topk(5) # 获取前5个最有可能的词

    # 打印预测结果
    print("Mask at position:", mask_index)
    print("Predictions:")
    for i, index_t in enumerate(predictions.indices):
        predicted_token = tokenizer.convert_ids_to_tokens([index_t])[0]
        print("Prediction {}:".format(i+1), predicted_token)
    print()
