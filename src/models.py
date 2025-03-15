"""模型定义和初始化"""

import numpy as np
from .config import FedConfig

class LinearModel:
    def __init__(self):
        """初始化线性模型"""
        self.weights = np.random.rand(FedConfig.INPUT_DIM)
    
    def predict(self, X):
        """模型预测
        
        Args:
            X: 输入特征，shape (n_samples, n_features)
            
        Returns:
            预测结果，shape (n_samples,)
        """
        return np.dot(X, self.weights)
    
    def get_weights(self):
        """获取模型参数"""
        return self.weights.copy()
    
    def set_weights(self, weights):
        """设置模型参数
        
        Args:
            weights: 新的模型参数
        """
        self.weights = weights.copy()
        
def create_model():
    """创建新模型实例"""
    return LinearModel()
