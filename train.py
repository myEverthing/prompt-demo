import argparse
import torch
import tqdm
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from torch.utils.data import DataLoader
from config import pretrained_mode, data_path
from get_data import get_dataset
from prompt_bert_model import PromptBertModel


# # 开始微调
# model.train()
# for epoch in range(3):  # 进行三个epoch的微调
#     for batch in train_loader:
#         input_ids, attention_mask = batch
#         optimizer.zero_grad()
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# # 保存微调后的模型
# model.save_pretrained("./prompt_finetuned_model")


def train(model, train_loader, test_loader, args):
    for epoch in range(args.epochs):
        model.train()
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch"):
            a = 1





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)
    parser.add_argument('--epochs', default=20, type=int)
    args = parser.parse_args()
    # 加载预训练模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_mode)
    # 定义训练数据
    train_dataset, test_dataset = get_dataset(data_path)
    # 定义训练集 DataLoader
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              drop_last=True, pin_memory=True)
    # 定义优化器和损失函数
    model = PromptBertModel(pretrained_mode)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()

    train(model, train_loader, test_loader, args)