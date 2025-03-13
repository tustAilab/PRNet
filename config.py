import os

# 数据集路径
DATASET_PATH = os.path.join(os.getcwd(), 'Dataset.csv')

# 空间关系文件路径
CONTACT_MAP_PATH = os.path.join(os.getcwd(), 'alignment_and_contacts_C1C2.pkl')

# 独热编码字母表
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AMINO_ACID_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
NUM_AMINO_ACIDS = len(AMINO_ACIDS)

# 训练集和验证集比例
TRAIN_RATIO = 0.8