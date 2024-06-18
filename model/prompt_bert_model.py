from transformers import BertTokenizer, BertForMaskedLM
import torch
import torch.nn as nn


label2ind = {
    "体育": [860, 5509],
    "娱乐": [2031, 727],
    "家居": [2157, 2233],
    "房产": [2791, 772],
    "教育": [3136, 5509],
    "时尚": [3198, 2213],
    "时政": [3198, 3124],
    "游戏": [3952, 2767],
    "社会": [4852, 833],
    "科技": [4906, 2825],
    "股票": [5500, 4873],
    "财经": [6568, 5307]
}

class PromptBertModel(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-chinese'):
        super(PromptBertModel, self).__init__()
        device = torch.device('cuda')
       
        self.bert = BertForMaskedLM.from_pretrained(pretrained_model_name)
        self.selected_values1 = torch.tensor([label2ind[label][0] for label in label2ind]).to(device)
        self.selected_values2 = torch.tensor([label2ind[label][1] for label in label2ind]).to(device)

    def forward(self, input_ids, mask_index, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        prediction_scores = outputs.logits  # 获取预测概率分布
      
        label1_tmp = torch.index_select(prediction_scores, dim=2, index=self.selected_values1)
        label2_tmp = torch.index_select(prediction_scores, dim=2, index=self.selected_values2)
        predict1 = torch.index_select(label1_tmp,dim = 1, index = mask_index[0])
        predict2 = torch.index_select(label2_tmp,dim = 1, index = mask_index[1])

        return predict1.squeeze(1),predict2.squeeze(1)