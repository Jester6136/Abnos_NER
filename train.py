from config.config import Config as config
from modules.dataset.NER_dataset import Dataset_NER,dataset_train

train_iter, eval_iter, tag2idx = dataset_train(config=config, bert_tokenizer=config.bert_model, do_lower_case=True,max_seq_length=512)

for batch in train_iter:
    # print(batch)  # Unpack the batch into inputs and labels (if applicable)
    for item in batch:
        a,b,c,d=item
        print(a)
        print(b)
        print(c)
        print(d)
    break
    