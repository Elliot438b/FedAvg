"""训练相关功能"""

import random
import numpy as np
from .config import FedConfig
from .utils import compute_loss, compute_gradient, aggregate_weights

def client_update(model, data, labels):
    """客户端本地训练
    
    Args:
        model: 模型实例
        data: 训练数据
        labels: 训练标签
        
    Returns:
        更新后的模型参数和平均损失
    """
    losses = []
    
    for _ in range(FedConfig.LOCAL_EPOCHS):
        # 前向传播
        predictions = model.predict(data)
        loss = compute_loss(labels, predictions)
        losses.append(loss)
        
        # 计算梯度并更新
        gradient = compute_gradient(data, labels, predictions)
        model.weights -= FedConfig.LEARNING_RATE * gradient
    
    return model.get_weights(), np.mean(losses)

def federated_training(model, client_data):
    """联邦学习训练过程
    
    Args:
        model: 全局模型实例
        client_data: 客户端数据字典，格式为 {client_id: (features, labels)}
        
    Returns:
        训练后的模型和训练历史
    """
    history = {'losses': []}
    
    for round in range(FedConfig.NUM_ROUNDS):
        # 随机选择客户端
        selected_clients = random.sample(
            range(FedConfig.NUM_CLIENTS),
            max(1, int(FedConfig.CLIENT_FRACTION * FedConfig.NUM_CLIENTS))
        )
        
        # 收集客户端更新
        client_weights = []
        client_losses = []
        sample_sizes = []
        
        for client in selected_clients:
            features, labels = client_data[client]
            
            # 创建客户端本地模型
            model.set_weights(model.get_weights())
            
            # 本地训练
            updated_weights, loss = client_update(model, features, labels)
            
            client_weights.append(updated_weights)
            client_losses.append(loss)
            sample_sizes.append(len(features))
        
        # 聚合模型参数
        aggregated_weights = aggregate_weights(client_weights, sample_sizes)
        model.set_weights(aggregated_weights)
        
        # 记录本轮平均损失
        avg_loss = np.mean(client_losses)
        history['losses'].append(avg_loss)
        
        print(f"Round {round+1}: Average Loss = {avg_loss:.6f}")
    
    print("\n训练完成！")
    print(f"初始损失值: {history['losses'][0]:.6f}")
    print(f"最终损失值: {history['losses'][-1]:.6f}")
    print(f"损失下降率: {((history['losses'][0] - history['losses'][-1]) / history['losses'][0] * 100):.2f}%")
    
    return model, history
