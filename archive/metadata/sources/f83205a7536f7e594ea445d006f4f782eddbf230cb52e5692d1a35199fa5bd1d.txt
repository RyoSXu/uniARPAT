import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# --- 配置参数 ---
# 建议根据你的数据集最大原子数设置，比如之前是 132 或 108
src_len = 82  # 原子最大数量
# edos 长度通常是 128，phdos 长度通常是 64
edos_len = 128
phdos_len = 64

pwd = "data/train4ARPAT"

def mkdirdt(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def csv2npy_unified(random_states=42):
    # 创建目录
    for s in ["train", "valid", "test"]:
        mkdirdt(os.path.join(pwd, s))

    # 1. 使用 names 列表处理不等长 CSV 读取
    # elements 只有一列原子类型，所以长度是 src_len
    names_el = [str(i) for i in range(src_len + 1)] 
    # position 每个原子 3 个坐标，所以长度是 src_len * 3
    names_pos = [str(i) for i in range(src_len * 3 + 1)]
    # DOS 数据通常是等长的，但也建议加上 names 以防万一
    names_edos = [str(i) for i in range(edos_len + 1)]
    names_phdos = [str(i) for i in range(phdos_len + 1)]

    print("Loading CSV files...")
    df_elements = pd.read_csv("data/elements.csv", names=names_el, header=None, index_col=0).fillna(0)
    df_position = pd.read_csv("data/position.csv", names=names_pos, header=None, index_col=0).fillna(0)
    df_edos     = pd.read_csv("data/edos.csv", names=names_edos, header=None, index_col=0).fillna(0)
    df_phdos    = pd.read_csv("data/phdos.csv", names=names_phdos, header=None, index_col=0).fillna(0)

    # 2. 取 ID 交集对齐所有数据
    common_ids = df_elements.index.intersection(df_position.index)\
                                 .intersection(df_edos.index)\
                                 .intersection(df_phdos.index)
    
    print(f"对齐后的共有材料数量: {len(common_ids)}")

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
        
        # 按照你之前的逻辑，逐个文件转 numpy 并保存
        # 保持文件名格式一致：{类别}_{划分}.npy
        curr_el = df_elements.iloc[idx].to_numpy()
        curr_pos = df_position.iloc[idx].to_numpy()
        curr_edos = df_edos.iloc[idx].to_numpy()
        curr_phdos = df_phdos.iloc[idx].to_numpy()
        curr_index = common_ids[idx].to_numpy()

        np.save(os.path.join(target_path, f"elements_{name}.npy"), curr_el)
        np.save(os.path.join(target_path, f"positions_{name}.npy"), curr_pos)
        np.save(os.path.join(target_path, f"edos_tgtdos_{name}.npy"), curr_edos)
        np.save(os.path.join(target_path, f"phdos_tgtdos_{name}.npy"), curr_phdos)
        np.save(os.path.join(target_path, f"{name}_index.npy"), curr_index)
        
        print(f"Done saving {name} set.")

if __name__ == "__main__":
    csv2npy_unified()