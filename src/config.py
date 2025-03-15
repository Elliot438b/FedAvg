"""联邦学习配置参数"""

class FedConfig:
    # 模型参数
    INPUT_DIM = 9  # 输入特征维度
    
    # 训练参数
    NUM_ROUNDS = 10        # 联邦学习轮数
    NUM_CLIENTS = 5        # 客户端数量
    CLIENT_FRACTION = 0.6  # 每轮参与训练的客户端比例
    LOCAL_EPOCHS = 5       # 本地训练轮数
    LEARNING_RATE = 0.1    # 学习率
    
    # 数据参数
    SAMPLES_PER_CLIENT = 100  # 每个客户端的样本数量
    
    # 随机种子
    RANDOM_SEED = 42
