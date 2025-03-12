import random
import numpy as np

def compute_loss(model, data):
    """计算模型在给定数据上的损失值（这里使用简单的均方误差）"""
    # 假设数据的最后一列是标签
    X = data[:, :-1]
    y = data[:, -1]
    predictions = np.dot(X, model)
    loss = np.mean((predictions - y) ** 2)
    return loss

def initialize_model():
    """初始化全局模型参数"""
    return np.random.rand(9)  # 改为9维，因为最后一列是标签

def client_update(model, data, epochs, lr):
    """客户端本地训练"""
    losses = []
    for _ in range(epochs):
        gradient = compute_gradient(model, data)
        model -= lr * gradient
        loss = compute_loss(model, data)
        losses.append(loss)
    return model, np.mean(losses)

def compute_gradient(model, data):
    """计算梯度（使用均方误差的梯度）"""
    X = data[:, :-1]
    y = data[:, -1]
    predictions = np.dot(X, model)
    gradient = -2 * np.dot(X.T, (y - predictions)) / len(data)
    return gradient

def aggregate_models(client_models, num_samples):
    """聚合客户端模型参数（加权平均）"""
    total_samples = sum(num_samples)
    weights = [n / total_samples for n in num_samples]
    new_global_model = np.zeros_like(client_models[0])
    for model, weight in zip(client_models, weights):
        new_global_model += model * weight
    return new_global_model

def federated_training(num_rounds, num_clients, fraction, local_epochs, lr):
    """联邦训练过程"""
    global_model = initialize_model()
    # 生成模拟数据：每个客户端100条数据，每条数据9个特征和1个标签
    client_data = {
        i: np.concatenate([
            np.random.rand(100, 9),  # 特征
            np.random.rand(100, 1)   # 标签
        ], axis=1) 
        for i in range(num_clients)
    }
    
    global_losses = []
    
    for round in range(num_rounds):
        selected_clients = random.sample(range(num_clients), max(1, int(fraction * num_clients)))
        client_models = []
        client_losses = []
        num_samples = []
        
        for client in selected_clients:
            local_model = global_model.copy()
            updated_model, local_loss = client_update(local_model, client_data[client], local_epochs, lr)
            client_models.append(updated_model)
            client_losses.append(local_loss)
            num_samples.append(len(client_data[client]))
        
        # 计算这一轮的平均损失
        avg_loss = np.mean(client_losses)
        global_losses.append(avg_loss)
        
        global_model = aggregate_models(client_models, num_samples)
        print(f"Round {round+1}: Average Loss = {avg_loss:.6f}")
    
    print("\n训练完成！")
    print(f"初始损失值: {global_losses[0]:.6f}")
    print(f"最终损失值: {global_losses[-1]:.6f}")
    print(f"损失下降率: {((global_losses[0] - global_losses[-1]) / global_losses[0] * 100):.2f}%")
    
    return global_model, global_losses

# 运行联邦学习
final_model, losses = federated_training(num_rounds=10, num_clients=5, fraction=0.6, local_epochs=5, lr=0.1)