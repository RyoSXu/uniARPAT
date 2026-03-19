import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def csv2npy_unified(pwd, random_states=42):
    # 1. 读取四个原始文件，设置第一列为 index (id)
    df_elements = pd.read_csv("data/edos_elements.csv", index_col=0)
    df_position = pd.read_csv("data/edos_position.csv", index_col=0)
    df_edos     = pd.read_csv("data/edos_tgtdos.csv", index_col=0)
    df_phdos    = pd.read_csv("data/phdos_tgtdos.csv", index_col=0)

    # 2. 取四个文件 ID 的交集
    common_ids = df_elements.index.intersection(df_position.index)\
                                 .intersection(df_edos.index)\
                                 .intersection(df_phdos.index)
    
    print(f"对齐后的共有材料数量: {len(common_ids)}")

    # 筛选数据
    df_elements = df_elements.loc[common_ids]
    df_position = df_position.loc[common_ids]
    df_edos     = df_edos.loc[common_ids]
    df_phdos    = df_phdos.loc[common_ids]

    # 3. 划分数据集 (8:1:1)
    indices = np.arange(len(common_ids))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=random_states)
    valid_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=random_states)

    splits = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    for name, idx in splits.items():
        target_path = os.path.join(pwd, name)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        
        # 保存结构
        np.save(os.path.join(target_path, f"elements_{name}.npy"), df_elements.iloc[idx].to_numpy())
        np.save(os.path.join(target_path, f"positions_{name}.npy"), df_position.iloc[idx].to_numpy())
        
        # 按照你的要求命名：edos_tgtdos 和 phdos_tgtdos
        np.save(os.path.join(target_path, f"edos_tgtdos_{name}.npy"), df_edos.iloc[idx].to_numpy())
        np.save(os.path.join(target_path, f"phdos_tgtdos_{name}.npy"), df_phdos.iloc[idx].to_numpy())
        
        print(f"已完成 {name} 文件夹数据保存。")

pwd = "data/train4ARPAT"
csv2npy_unified(pwd)