import torch
from transformers import BertTokenizer, BertForMaskedLM
from config import pretrained_mode, data_path, TEMPLATE
import pandas as pd
import re
# 加载BERT tokenizer和预训练模型
tokenizer = BertTokenizer.from_pretrained("./pretrained_model/bert-base-chinese")
model = BertForMaskedLM.from_pretrained("./pretrained_model/bert-base-chinese")

def template_process(processed_text, template):
 
    template_text = template.replace('*mask*', tokenizer.mask_token)\
                               .replace('*sep+*', '')\
                               .replace('*cls*', '').replace('*sent_0*', processed_text).replace('_', '')

    return template_text

def padding_process(text, padding_strategy = "head_only", max_length = 510):
        if padding_strategy == 'head_only':
            return text[:max_length]
        elif padding_strategy == 'tail_only':
            return text[-max_length:]
        elif padding_strategy == 'head_and_tail':
            head_length = 128
            tail_length = max_length - head_length
            return text[:head_length] + text[-tail_length:]
        else:
            raise ValueError("Invalid padding strategy. Please choose from 'head_only', 'tail_only', or 'head_and_tail'.")

def preprocessed(text):
    modified_text = re.sub(r"\xa0", "", text)
    return modified_text

docs = pd.read_csv(data_path)

random_row = docs.sample(n=1)

text = random_row["NEWSCONTENT"].tolist()
type = random_row["NEWSTYPE"].tolist()

ori_text = preprocessed(text[0])

padding_text = padding_process(ori_text,padding_strategy = "head_only", max_length = 510)
template_test = template_process(padding_text, TEMPLATE)

print(template_test)