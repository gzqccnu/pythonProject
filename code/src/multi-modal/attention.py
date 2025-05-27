## 12. 系统训练和使用示例

# 导入需要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2
from ultralytics import YOLO  # 导入YOLOv8
import math
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50

# 系统初始化示例
def initialize_system(yolo_weights_path="best.pt"):
    """
    初始化油茶花期识别系统
    
    Args:
        yolo_weights_path: 预训练的YOLOv8模型权重路径
    
    Returns:
        初始化好的多模态系统
    """
    # 创建多模态系统
    system = OilTeaFlowerMultiModalSystem(yolo_model_path=yolo_weights_path)
    
    # 加载其他模型权重（如果有）
    # system.load_state_dict(torch.load('multimodal_system.pth'))
    
    return system

# YOLOv8单独训练示例

# 系统使用示例
def run_system_demo(image_path, gas_data, vis=50, hum=80):
    """
    系统运行演示
    
    Args:
        image_path: 输入图像路径
        gas_data: 气态数据
        vis: 能见度
        hum: 湿度
    """
    # 加载系统
    system = initialize_system()
    system.eval()
    
    # 准备输入数据
    # 1. 视觉输入
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = prepare_image(image)  # 图像预处理函数
    
    # 2. 气态输入
    gas_tensor = torch.tensor(gas_data, dtype=torch.float32).unsqueeze(0)
    
    # 3. 环境条件
    visibility = torch.tensor([[vis]], dtype=torch.float32)
    humidity = torch.tensor([[hum]], dtype=torch.float32)
    
    # 运行推理
    with torch.no_grad():
        result = inference(system, image_tensor, gas_tensor, visibility, humidity)
    
    # 显示结果
    print(f"当前油茶花期: {result['flower_stage']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"视觉权重: {result['visual_weight']:.2f}")
    print(f"气态权重: {result['gas_weight']:.2f}")
    
    # 如果有检测可视化结果，显示它
    if 'detection_visualization' in result:
        cv2.imshow("Oil Tea Flower Detection", result['detection_visualization'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result

# 图像预处理函数
def prepare_image(image, size=640):
    """将图像预处理为模型输入格式"""
    # 调整图像大小
    img = cv2.resize(image, (size, size))
    # 归一化
    img = img / 255.0
    # 转为张量
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
    # 添加批次维度
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

# 示例调用
if __name__ == "__main__":
    
    # 2. 运行系统演示
    # 示例气态数据：温度, 湿度, 光照, VOC1, VOC2, VOC3, VOC4, 气压
    gas_data = np.array([[22.5, 75.3, 850, 0.12, 0.08, 0.22, 0.05, 1013.2]])
    
    run_system_demo(
        image_path= r"F:\File\Camellia oleifera\Camellia_oleifera\1.jpg",
        gas_data=gas_data,
        vis=80,  # 能见度
        hum=75   # 湿度
    )
    
# 油茶双模态花期识别系统多模态实现方案

## 1. 多模态特征提取

### 1.1 视觉模态特征提取

class VisualFeatureExtractor(nn.Module):
    def __init__(self, yolo_model_path=None):
        super(VisualFeatureExtractor, self).__init__()
        # 直接加载您的YOLOv8模型
        if yolo_model_path:
            # 方式一：加载已训练好的YOLOv8模型
            self.yolo_model = YOLO(yolo_model_path)  # 从指定路径加载您的YOLOv8模型
        else:
            # 方式二：初始化YOLOv8模型（如果不加载预训练模型）
            self.yolo_model = YOLO('yolov8s.pt')  # 或使用其他YOLOv8变体
            
        # 多尺度特征提取器
        self.feature_extractor = YOLOFeatureExtractor(self.yolo_model)
        
        # 多尺度特征融合
        self.neck = PANet(in_channels=[256, 512, 1024])  # 调整通道数以匹配YOLOv8输出
        
        # 特征压缩模块，将视觉特征压缩到统一维度
        self.compress = nn.Conv2d(512, 256, kernel_size=1)
        
    def forward(self, x):
        # 提取YOLOv8多尺度特征
        features = self.feature_extractor(x)
        
        # 融合多尺度特征
        fused_features = self.neck(features)
        
        # 压缩特征维度
        compressed_features = self.compress(fused_features)
        
        return compressed_features, features

# YOLOv8特征提取器
class YOLOFeatureExtractor:
    def __init__(self, yolo_model):
        self.yolo_model = yolo_model
        self.hooks = []
        self.features = []
        
        # 注册钩子以获取中间特征图
        self._register_hooks()
        
    def _register_hooks(self):
        # 根据YOLOv8模型结构注册提取多尺度特征的钩子
        # 以下是示例，实际钩子位置需要根据您的YOLOv8具体结构调整
        model = self.yolo_model.model
        
        # 清除之前的钩子
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # 定义提取特征的层
        # 注意：以下层索引需要根据您的YOLOv8模型结构进行适配
        feature_layers = [6, 10, 14]  # 例如：浅层、中层、深层特征
        
        def hook_fn(module, input, output):
            self.features.append(output)
            
        for idx in feature_layers:
            if hasattr(model, 'model') and idx < len(model.model):
                hook = model.model[idx].register_forward_hook(hook_fn)
                self.hooks.append(hook)
    
    def __call__(self, x):
        # 清除之前的特征
        self.features = []
        
        # 推理但不执行检测后处理（只提取特征）
        with torch.no_grad():
            _ = self.yolo_model.model(x, augment=False)
            
        # 返回提取的多尺度特征
        return self.features

### 1.2 气态模态特征提取


class GasFeatureExtractor(nn.Module):
    def __init__(self, input_dim=8):
        super(GasFeatureExtractor, self).__init__()
        # 输入特征: 温度、湿度、光照、VOCs浓度等
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        # 自注意力机制强化关键时间节点
        self.self_attention = SelfAttention(hidden_size=256)
        # 特征映射
        self.projection = nn.Linear(256, 256)
        
    def forward(self, x):
        # x: (batch_size, seq_len, features)
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, 256)
        # 应用自注意力机制
        attended_features = self.self_attention(lstm_out)
        # 压缩时间维度，只保留最后一个时间步
        last_features = attended_features[:, -1, :]
        # 投影到统一特征空间
        gas_features = self.projection(last_features)
        return gas_features

## 2. 协同注意力机制

### 2.1 自注意力模块

   
class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = math.sqrt(hidden_size)
        
    def forward(self, x):
        # x: (batch_size, seq_len, hidden_size)
        q = self.query(x)  # (batch_size, seq_len, hidden_size)
        k = self.key(x)    # (batch_size, seq_len, hidden_size)
        v = self.value(x)  # (batch_size, seq_len, hidden_size)
        
        # 计算注意力分数
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 加权求和
        context = torch.matmul(attention_weights, v)
        return context
 

### 2.2 协同注意力矩阵构建

   
class CooperativeAttention(nn.Module):
    def __init__(self, visual_dim=256, gas_dim=256):
        super(CooperativeAttention, self).__init__()
        # 视觉→气态路径
        self.visual_to_gas = nn.Sequential(
            nn.Conv2d(visual_dim, gas_dim, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 气态→视觉路径
        self.gas_to_visual = nn.Sequential(
            nn.Linear(gas_dim, visual_dim),
            nn.Sigmoid()
        )
        
    def forward(self, visual_features, gas_features):
        # visual_features: (batch_size, channels, height, width)
        # gas_features: (batch_size, gas_dim)
        
        # 视觉→气态：生成空间注意力图
        batch_size = visual_features.size(0)
        spatial_attn = self.visual_to_gas(visual_features)  # (batch_size, gas_dim, 1, 1)
        spatial_attn = spatial_attn.view(batch_size, -1)    # (batch_size, gas_dim)
        
        # 注意力加权的气态特征
        weighted_gas = gas_features * spatial_attn
        
        # 气态→视觉：生成通道注意力掩膜
        channel_attn = self.gas_to_visual(gas_features)     # (batch_size, visual_dim)
        channel_attn = channel_attn.unsqueeze(-1).unsqueeze(-1)  # (batch_size, visual_dim, 1, 1)
        
        # 注意力加权的视觉特征
        weighted_visual = visual_features * channel_attn
        
        return weighted_visual, weighted_gas
 

## 3. 动态权重分配

   
class DynamicWeightController(nn.Module):
    def __init__(self):
        super(DynamicWeightController, self).__init__()
        # 输入：能见度、湿度
        self.mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        # 上一帧权重
        self.prev_weight = 0.5
        # 最大权重变化幅度
        self.max_change = 0.2
        
    def forward(self, visibility, humidity):
        # 拼接能见度和湿度
        env_condition = torch.cat([visibility, humidity], dim=1)
        # 计算当前权重
        current_weight = self.mlp(env_condition)
        
        # 应用软阈值约束，防止权重突变
        weight_diff = current_weight - self.prev_weight
        clamped_diff = torch.clamp(weight_diff, -self.max_change, self.max_change)
        smooth_weight = self.prev_weight + clamped_diff
        
        # 更新上一帧权重
        self.prev_weight = smooth_weight.item()
        
        # 根据当前条件设置默认权重（规则）
        mask_poor_visibility = (visibility < 50) & (humidity > 80)
        mask_clear_condition = visibility > 1000
        
        # 根据条件强制覆盖权重
        final_weight = smooth_weight.clone()
        final_weight[mask_poor_visibility] = 0.2  # 恶劣天气条件下，视觉权重降低
        final_weight[mask_clear_condition] = 0.85  # 晴朗条件下，视觉权重固定为0.85
        
        return final_weight
 

## 4. 特征融合模块

   
class ModalityFusion(nn.Module):
    def __init__(self, feature_dim=256):
        super(ModalityFusion, self).__init__()
        # 混合维度融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 128)
        )
        # 残差连接
        self.residual_conn = nn.Linear(feature_dim * 2, 128)
        
    def forward(self, visual_features, gas_features, alpha):
        # 压缩视觉特征的空间维度
        visual_vec = visual_features.mean(dim=[2, 3])  # (batch_size, visual_dim)
        
        # 加权融合
        fusion = alpha * visual_vec + (1 - alpha) * gas_features
        
        # 拼接特征以保留原始模态信息
        concat_features = torch.cat([visual_vec, gas_features], dim=1)
        
        # 通过融合层
        fused_features = self.fusion_layer(concat_features)
        
        # 残差连接
        residual = self.residual_conn(concat_features)
        
        # 最终融合特征
        final_features = fused_features + residual
        
        return final_features
 

## 5. 模态一致性验证

   
class ModalConsistencyVerification(nn.Module):
    def __init__(self, feature_dim=128):
        super(ModalConsistencyVerification, self).__init__()
        # 独立分类器
        self.visual_classifier = nn.Linear(feature_dim, 12)  # 12个花期等级
        self.gas_classifier = nn.Linear(feature_dim, 12)
        self.fusion_classifier = nn.Linear(feature_dim, 12)
        
        # 贝叶斯一致性评估
        self.consistency_network = BayesianNetwork(12)
        
    def forward(self, visual_features, gas_features, fused_features):
        # 各模态独立预测
        visual_pred = self.visual_classifier(visual_features)
        gas_pred = self.gas_classifier(gas_features)
        fusion_pred = self.fusion_classifier(fused_features)
        
        # 贝叶斯一致性评估
        confidence_scores = self.consistency_network(visual_pred, gas_pred, fusion_pred)
        
        # 根据一致性评分调整最终预测
        final_pred = fusion_pred * confidence_scores
        
        return final_pred, confidence_scores
 

### 5.1 贝叶斯一致性网络

   
class BayesianNetwork(nn.Module):
    def __init__(self, num_classes):
        super(BayesianNetwork, self).__init__()
        self.num_classes = num_classes
        # 一致性评估网络
        self.consistency_net = nn.Sequential(
            nn.Linear(num_classes * 3, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, visual_pred, gas_pred, fusion_pred):
        # 计算模态间的预测差异
        visual_probs = F.softmax(visual_pred, dim=1)
        gas_probs = F.softmax(gas_pred, dim=1)
        fusion_probs = F.softmax(fusion_pred, dim=1)
        
        # 拼接所有预测结果
        all_preds = torch.cat([visual_probs, gas_probs, fusion_probs], dim=1)
        
        # 计算一致性评分
        confidence_scores = self.consistency_net(all_preds)
        
        return confidence_scores
 

## 6. 动态反馈调整机制

   
class DynamicFeedbackController(nn.Module):
    def __init__(self, feature_dim=128):
        super(DynamicFeedbackController, self).__init__()
        # 任务性能评估
        self.performance_evaluator = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 输出两个通道：视觉和气态模态的反馈系数
        )
        
    def forward(self, fused_features, loss_value):
        # 根据融合特征和损失值评估当前性能
        feedback_coeffs = self.performance_evaluator(fused_features)
        
        # 归一化反馈系数
        normalized_coeffs = F.softmax(feedback_coeffs, dim=1)
        
        # 将损失加权分配给两个模态
        visual_feedback = normalized_coeffs[:, 0] * loss_value
        gas_feedback = normalized_coeffs[:, 1] * loss_value
        
        return visual_feedback, gas_feedback
 

## 7. 完整多模态系统集成

   
class OilTeaFlowerMultiModalSystem(nn.Module):
    def __init__(self, yolo_model_path=None):
        super(OilTeaFlowerMultiModalSystem, self).__init__()
        # 视觉特征提取（整合YOLOv8）
        self.visual_extractor = VisualFeatureExtractor(yolo_model_path)
        # 气态特征提取
        self.gas_extractor = GasFeatureExtractor()
        # 协同注意力机制
        self.cooperative_attention = CooperativeAttention()
        # 动态权重控制器
        self.weight_controller = DynamicWeightController()
        # 模态融合模块
        self.fusion_module = ModalityFusion()
        # 模态一致性验证
        self.consistency_verification = ModalConsistencyVerification()
        # 动态反馈控制器
        self.feedback_controller = DynamicFeedbackController()
        # 花期分类器
        self.flower_classifier = nn.Linear(128, 12)  # 12个花期等级
        
    def forward(self, visual_input, gas_input, visibility, humidity):
        # 1. 特征提取
        visual_features, _ = self.visual_extractor(visual_input)
        gas_features = self.gas_extractor(gas_input)
        
        # 2. 协同注意力
        weighted_visual, weighted_gas = self.cooperative_attention(
            visual_features, gas_features
        )
        
        # 3. 动态权重分配
        alpha = self.weight_controller(visibility, humidity)
        
        # 4. 特征融合
        fused_features = self.fusion_module(
            weighted_visual, weighted_gas, alpha
        )
        
        # 5. 最终花期分类
        flower_stage_logits = self.flower_classifier(fused_features)
        
        # 6. 模态一致性验证
        final_pred, confidence = self.consistency_verification(
            weighted_visual.mean(dim=[2, 3]), 
            weighted_gas, 
            fused_features
        )
        
        return {
            'flower_stage': final_pred,
            'confidence': confidence,
            'modality_weight': alpha,
            'visual_features': weighted_visual,
            'gas_features': weighted_gas,
            'fused_features': fused_features
        }
    
    def get_feedback(self, loss_value, fused_features):
        # 计算反馈系数
        visual_feedback, gas_feedback = self.feedback_controller(
            fused_features, loss_value
        )
        return visual_feedback, gas_feedback
    
    # 添加独立的YOLOv8检测方法，用于直接获取油茶花检测结果
    def detect_flowers(self, image):
        """使用YOLOv8直接检测图像中的油茶花"""
        if hasattr(self.visual_extractor, 'yolo_model'):
            # 调用YOLOv8模型进行检测
            results = self.visual_extractor.yolo_model(image)
            return results
        else:
            raise ValueError("YOLOv8模型未正确加载到视觉特征提取器中")
 

## 8. 训练流程

   
def train_multimodal_system(model, train_loader, optimizer, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            # 获取输入数据
            visual_input = batch['visual'].to(device)
            gas_input = batch['gas'].to(device)
            visibility = batch['visibility'].to(device)
            humidity = batch['humidity'].to(device)
            labels = batch['flower_stage'].to(device)
            
            # 前向传播
            outputs = model(visual_input, gas_input, visibility, humidity)
            flower_pred = outputs['flower_stage']
            
            # 计算损失
            loss = criterion(flower_pred, labels)
            total_loss += loss.item()
            
            # 反向传播前清零梯度
            optimizer.zero_grad()
            loss.backward()
            
            # 获取反馈系数
            visual_feedback, gas_feedback = model.get_feedback(
                loss.item(), outputs['fused_features'].detach()
            )
            
            # 应用反馈 - 这里简化处理，实际中需要更复杂的反馈机制
            # 例如可以根据反馈调整学习率或特征提取权重
            
            # 更新参数
            optimizer.step()
            
        # 打印每个epoch的平均损失
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')
 

## 9. 推理流程

   
def inference(model, visual_input, gas_input, visibility, humidity):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        # 转移数据到设备
        visual_input = visual_input.to(device)
        gas_input = gas_input.to(device)
        visibility = visibility.to(device)
        humidity = humidity.to(device)
        
        # 模型推理
        outputs = model(visual_input, gas_input, visibility, humidity)
        
        # 获取花期预测和置信度
        flower_stage_pred = outputs['flower_stage']
        confidence = outputs['confidence']
        modality_weight = outputs['modality_weight']
        
        # 解码花期阶段
        flower_stage_idx = torch.argmax(flower_stage_pred, dim=1).item()
        flower_stages = [
            '萌芽期-早', '萌芽期-中', '萌芽期-晚',
            '初花期-早', '初花期-中', '初花期-晚',
            '盛花期-早', '盛花期-中', '盛花期-晚',
            '末花期-早', '末花期-中', '末花期-晚'
        ]
        
        result = {
            'flower_stage': flower_stages[flower_stage_idx],
            'confidence': confidence[0][flower_stage_idx].item(),
            'visual_weight': modality_weight.item(),
            'gas_weight': 1 - modality_weight.item()
        }
        
        # YOLO检测结果可视化（如果需要）
        if hasattr(model, 'visual_extractor') and hasattr(model.visual_extractor, 'yolo_model'):
            # 使用YOLOv8进行花朵检测
            yolo_results = model.visual_extractor.yolo_model(visual_input.cpu().numpy())
            # 绘制检测结果
            annotated_frame = yolo_results[0].plot()
            result['detection_visualization'] = annotated_frame
        
        return result
 

## 10. 冲突检测与消解机制

   
class ConflictDetectionResolver:
    def __init__(self, num_classes=12, threshold=0.3):
        self.num_classes = num_classes
        self.threshold = threshold
        
    def detect_conflict(self, visual_pred, gas_pred):
        # 获取各模态的最高预测类别
        visual_class = torch.argmax(visual_pred, dim=1)
        gas_class = torch.argmax(gas_pred, dim=1)
        
        # 检测预测是否一致
        is_conflict = visual_class != gas_class
        
        # 计算预测差异程度
        visual_conf = torch.max(F.softmax(visual_pred, dim=1), dim=1)[0]
        gas_conf = torch.max(F.softmax(gas_pred, dim=1), dim=1)[0]
        
        # 差异程度超过阈值则视为冲突
        significant_conflict = is_conflict & (torch.abs(visual_conf - gas_conf) > self.threshold)
        
        return significant_conflict
    
    def resolve_conflict(self, visual_pred, gas_pred, env_condition):
        # 检测冲突
        conflicts = self.detect_conflict(visual_pred, gas_pred)
        
        if not torch.any(conflicts):
            # 无冲突，返回平均预测
            return (visual_pred + gas_pred) / 2
        
        # 冲突解决方法：基于环境条件的加权投票
        # 例如，在低能见度条件下，更信任气态预测
        visual_weight = torch.ones_like(conflicts, dtype=torch.float32)
        gas_weight = torch.ones_like(conflicts, dtype=torch.float32)
        
        # 根据环境条件调整权重
        visibility = env_condition[:, 0]  # 假设第一个参数是能见度
        humidity = env_condition[:, 1]    # 假设第二个参数是湿度
        
        # 低能见度高湿度条件，降低视觉权重
        bad_conditions = (visibility < 50) & (humidity > 80)
        visual_weight[bad_conditions] = 0.3
        gas_weight[bad_conditions] = 0.7
        
        # 好天气条件，提高视觉权重
        good_conditions = visibility > 1000
        visual_weight[good_conditions] = 0.7
        gas_weight[good_conditions] = 0.3
        
        # 加权融合
        visual_weight = visual_weight.unsqueeze(1).expand_as(visual_pred)
        gas_weight = gas_weight.unsqueeze(1).expand_as(gas_pred)
        
        resolved_pred = visual_weight * visual_pred + gas_weight * gas_pred
        
        return resolved_pred
 

## 11. 完整多模态系统评估

   
def evaluate_multimodal_system(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 初始化评估指标
    correct = 0
    total = 0
    visual_only_correct = 0
    gas_only_correct = 0
    
    # 天气条件分类评估
    clear_correct = 0
    clear_total = 0
    rainy_correct = 0
    rainy_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            # 获取输入数据
            visual_input = batch['visual'].to(device)
            gas_input = batch['gas'].to(device)
            visibility = batch['visibility'].to(device)
            humidity = batch['humidity'].to(device)
            labels = batch['flower_stage'].to(device)
            
            # 模型推理 - 完整多模态系统
            outputs = model(visual_input, gas_input, visibility, humidity)
            flower_pred = outputs['flower_stage']
            predicted = torch.argmax(flower_pred, dim=1)
            
            # 仅视觉模态
            visual_features, _ = model.visual_extractor(visual_input) 
            visual_only_pred = model.flower_classifier(visual_features.mean(dim=[2, 3]))
            visual_predicted = torch.argmax(visual_only_pred, dim=1)
            
            # 仅气态模态
            gas_features = model.gas_extractor(gas_input)
            gas_only_pred = model.flower_classifier(gas_features)
            gas_predicted = torch.argmax(gas_only_pred, dim=1)
            
            # 统计正确预测数
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            visual_only_correct += (visual_predicted == labels).sum().item()
            gas_only_correct += (gas_predicted == labels).sum().item()
            
            # 按天气条件分类评估
            clear_mask = visibility > 1000
            rainy_mask = (visibility < 200) & (humidity > 80)
            
            # 晴好天气评估
            if torch.any(clear_mask):
                clear_labels = labels[clear_mask]
                clear_pred = predicted[clear_mask]
                clear_total += clear_labels.size(0)
                clear_correct += (clear_pred == clear_labels).sum().item()
                
            # 雨雾天气评估
            if torch.any(rainy_mask):
                rainy_labels = labels[rainy_mask]
                rainy_pred = predicted[rainy_mask]
                rainy_total += rainy_labels.size(0)
                rainy_correct += (rainy_pred == rainy_labels).sum().item()
    
    # 计算准确率
    accuracy = correct / total * 100
    visual_only_accuracy = visual_only_correct / total * 100
    gas_only_accuracy = gas_only_correct / total * 100
    
    # 计算不同天气条件下的准确率
    clear_accuracy = clear_correct / clear_total * 100 if clear_total > 0 else 0
    rainy_accuracy = rainy_correct / rainy_total * 100 if rainy_total > 0 else 0
    
    # 输出评估结果
    print(f'总体准确率: {accuracy:.2f}%')
    print(f'仅视觉模态准确率: {visual_only_accuracy:.2f}%')
    print(f'仅气态模态准确率: {gas_only_accuracy:.2f}%')
    print(f'晴好天气准确率: {clear_accuracy:.2f}%')
    print(f'雨雾天气准确率: {rainy_accuracy:.2f}%')
    
    return {
        'overall_accuracy': accuracy,
        'visual_only_accuracy': visual_only_accuracy,
        'gas_only_accuracy': gas_only_accuracy,
        'clear_weather_accuracy': clear_accuracy,
        'rainy_weather_accuracy': rainy_accuracy
    }
