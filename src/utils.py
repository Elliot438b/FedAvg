"""工具函数"""

import numpy as np

def compute_loss(y_true, y_pred):
    """计算均方误差损失
    
    Args:
        y_true: 真实标签
        y_pred: 预测值
        
    Returns:
        损失值
    """
    return np.mean((y_true - y_pred) ** 2)

def compute_gradient(X, y_true, y_pred):
    """计算梯度
    
    Args:
        X: 输入特征
        y_true: 真实标签
        y_pred: 预测值
        
    Returns:
        梯度
    """
    return -2 * np.dot(X.T, (y_true - y_pred)) / len(X)

def aggregate_weights(weights_list, sample_sizes):
    """聚合模型参数（加权平均）
    
    Args:
        weights_list: 客户端模型参数列表
        sample_sizes: 对应的样本数量列表
        
    Returns:
        聚合后的模型参数
    """
    total_samples = sum(sample_sizes)
    weights = [n / total_samples for n in sample_sizes]
    
    aggregated = np.zeros_like(weights_list[0])
    for w, weight in zip(weights_list, weights):
        aggregated += w * weight
        
    return aggregated
