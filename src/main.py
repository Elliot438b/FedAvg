"""主程序入口"""

import numpy as np
from .config import FedConfig
from .models import create_model
from .training import federated_training

def generate_data():
    """生成模拟数据
    
    Returns:
        client_data: 客户端数据字典，格式为 {client_id: (features, labels)}
    """
    np.random.seed(FedConfig.RANDOM_SEED)
    
    client_data = {}
    for i in range(FedConfig.NUM_CLIENTS):
        # 生成特征
        features = np.random.rand(FedConfig.SAMPLES_PER_CLIENT, FedConfig.INPUT_DIM)
        # 生成标签（这里使用简单的线性关系加噪声）
        true_weights = np.random.rand(FedConfig.INPUT_DIM)
        labels = np.dot(features, true_weights) + np.random.normal(0, 0.1, FedConfig.SAMPLES_PER_CLIENT)
        
        client_data[i] = (features, labels)
    
    return client_data

def main():
    """主函数"""
    # 生成数据
    client_data = generate_data()
    
    # 创建模型
    model = create_model()
    
    # 开始训练
    print("开始联邦学习训练...")
    trained_model, history = federated_training(model, client_data)
    
    return trained_model, history

if __name__ == "__main__":
    main()
