from datasets import Dataset
from datasets import load_from_disk
import pickle
import glob
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
"./lkm_ckpt0909/LKM1_32M",
trust_remote_code=True)
def get_ids(df,max_length=27,stride=22):
    token_ids = tokenizer.encode_df(df)
   
    result_list = []
    input_ids = token_ids[:-1]
    labels_ids = token_ids[1:]
    for i in range(0, len(token_ids) - max_length, stride):
        
        input_chunk = input_ids[i:i + max_length]
        label_chunk = labels_ids[i:i + max_length]
        assert len(input_chunk)==len(label_chunk)
        data = {
            "input_ids": input_chunk,
            "label_ids": label_chunk,
            "tokens":len(input_chunk)
        }
        result_list.append(data)
    return result_list
# 创建 Dataset 对象
def norm_jk_df(df):
    df = df.dropna()
    df = df.reset_index()
    df = df.rename(columns={'index': 'date'})
    df =df[['date','open','high','low','close']]
    return df
def get_datasets():
    data_pkl = glob.glob('data/*.pkl')
    all_result_list = []
    for data in tqdm(data_pkl):
        with open(data,'rb') as fr:
            df_list = pickle.load(fr)
        for df in df_list:
            df = df.iloc[:-5]
            
            df =norm_jk_df(df)
            print(df.tail(5))
            try:
                result_list = get_ids(df)
            except:
                continue
            all_result_list.extend(result_list)
    total_tokens = 0
    for chunk in all_result_list:
        total_tokens += chunk['tokens']
    print(f'共有chunk:{len(all_result_list)}，tokens:{total_tokens}')
    
    dataset = Dataset.from_list(all_result_list)
            
       
 

    # 拆分为训练集和验证集（90% / 10%）
    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)

    # 获取训练集和验证集
    train_dataset = dataset_split["train"]
    val_dataset = dataset_split["test"]

    # 保存到磁盘
    train_dataset.save_to_disk("./train_data/train_dataset3")
    val_dataset.save_to_disk("./train_data/val_dataset3")

    # 载入数据集
    train_dataset_loaded = load_from_disk("./train_data/train_dataset3")
    val_dataset_loaded = load_from_disk("./train_data/val_dataset3")

    # 打印检查
    print(train_dataset_loaded[0])
    print(val_dataset_loaded[0])


if __name__=='__main__':
    get_datasets()