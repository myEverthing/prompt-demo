import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
class NewsDataset(Dataset):
    def __init__(self, data, labels, template, tokenizer, padding_strategy="head-only", max_length = 510):
        self.data = data
        self.labels = labels
        self.padding_strategy = padding_strategy
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.template = template
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        # 根据索引获取样本和标签
        ori_text = self.data[index]
        label = self.labels[index]
        ###
        #  数据预处理
        #  padding 处理  
        #  1、head-only：  保留前510
        #  2、tail-only: keep the last 510 tokens
        #  3、head+tail: empirically select the first 128 and the last 382 tokens
        processed_text  = self.padding_process(ori_text, self.padding_strategy, self.max_length)
        #  添加模板       
        template_text = self.template_process(self, processed_text, self.template)
        # 编码
        text_input_id, attention_mask = self.get_tokenized(template_text)
        # 在这里你可能还会进行一些预处理，如数据转换等
        return text_input_id, attention_mask, label
    
    def padding_process(text, padding_strategy = "head-only", max_length = 510):
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


    def template_process(self, processed_text, template):
 
        template_text = template.replace('*mask*', self.tokenizer.mask_token)\
                               .replace('*sep+*', '')\
                               .replace('*cls*', '').replace('*sent_0*', processed_text).replace('_', '')
        return template_text
    
    def get_tokenized(self, text):
        """
        对单个文档的文本text做tokenizer
        :param text: 纯文本内容
        :return:
        """
        tokens = self.tokenizer.tokenize(text)
        # 加上起始标志
        tokens = ["[CLS]"] + tokens
    
        # 把token 转换成id， encode_result是转换成id后的数字列表
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)

        #如果小于最长长度，那么做padding
        padding = [0] * (self.max_seq_len + 2 - len(encode_result))

        #padding 0加到末尾
        encode_result += padding
        attention_mask = [1] * len(tokens) + [0] * (self.max_seq_len + 2 - len(tokens))
        return torch.tensor(encode_result), torch.tensor(attention_mask)