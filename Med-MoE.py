import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class MedMoE(nn.Module):
    """
    Med-MoE: 多模态医学视觉-语言模型
    架构对应您的流程图描述
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # === Layer 1: 输入层 ===
        # (隐式: 图像和文本输入)
        
        # === Layer 2: 核心处理层 ===
        # Stage 1: 视觉感知与特征增强
        self.visual_encoder = VisualEncoder(config)      # 模块1
        self.sam3_engine = SAM3Engine(config)            # 模块2
        self.feature_fusion = MultiModalFusion(config)   # 模块3
        
        # Stage 2: 专家混合决策与推理
        self.moe_router = MOERouter(config)              # 模块4
        self.expert_pool = ExpertPool(config)            # 模块5
        self.expert_aggregation = ExpertAggregation(config) # 模块6
        
        # Stage 3: 像素级接地与生成
        self.llm_core = LLMCore(config)                  # 模块7
        self.grounding_head = GroundingHead(config)      # 模块8
        self.text_decoder = TextDecoder(config)          # 模块9
        
        # === Layer 3: 输出层 ===
        self.output_projector = OutputProjector(config)
        
    def forward(self, image, text_instruction):
        """
        前向传播流程
        Args:
            image: (B, 3, H, W) 医学图像
            text_instruction: List[str] 文本指令
        Returns:
            text_report: 文本报告
            segmentation_mask: 分割掩码
            expert_weights: 专家权重（可解释性）
        """
        B = image.shape[0]
        
        # ===== Stage 1: 视觉感知与特征增强 =====
        # 模块1: ViT编码器 (带DoRA)
        vit_features = self.visual_encoder(image)  # (B, N, D_vit)
        
        # 模块2: SAM 3引擎 (初始无提示)
        # 第一次前向: 使用默认提示或无提示
        initial_sam_output = self.sam3_engine(
            image, 
            prompts=None,  # 初始无提示
            return_features=True
        )
        initial_masks = initial_sam_output['masks']  # (B, 3, H, W)
        sam_features = initial_sam_output['features']  # (B, D_sam, H', W')
        
        # 模块3: 特征融合
        # 将ViT特征和SAM特征融合
        fused_visual_tokens = self.feature_fusion(
            vit_features, 
            sam_features
        )  # (B, N, D_fused)
        
        # ===== Stage 2: 专家混合决策与推理 =====
        # 编码文本指令
        text_embeddings = self.text_encoder(text_instruction)  # (B, L, D_text)
        text_global = text_embeddings.mean(dim=1)  # (B, D_text)
        
        # 模块4: MoE路由网络
        # 结合视觉和文本特征进行路由决策
        router_input = torch.cat([
            fused_visual_tokens.mean(dim=1),  # 全局视觉特征 (B, D_fused)
            text_global  # 全局文本特征 (B, D_text)
        ], dim=-1)
        
        expert_gates = self.moe_router(router_input)  # (B, num_experts)
        
        # 模块5: 专家池 + 模块6: 专家聚合
        moe_output = self.expert_pool(
            fused_visual_tokens, 
            expert_gates,
            text_embeddings
        )  # (B, D_moe)
        
        # ===== Stage 3: 像素级接地与生成 =====
        # 模块7: LLM核心
        llm_hidden_states = self.llm_core(
            input_embeds=moe_output.unsqueeze(1),  # (B, 1, D_moe)
            text_embeddings=text_embeddings
        )  # (B, L_seq, D_llm)
        
        # 模块8: 接地模块（关键！从LLM生成SAM提示）
        # 从LLM的中间层提取空间意图
        spatial_intent = self.grounding_head(llm_hidden_states)  # (B, D_prompt)
        
        # 将空间意图解码为SAM 3提示
        sam_prompts = self.decode_spatial_intent(spatial_intent)  # 点/框坐标
        
        # === 反馈回路: 用生成的提示重新运行SAM 3 ===
        # 这是架构的关键创新点！
        refined_sam_output = self.sam3_engine(
            image,
            prompts=sam_prompts,  # 使用LLM生成的提示
            return_features=True
        )
        refined_masks = refined_sam_output['masks']
        refined_features = refined_sam_output['features']
        
        # 模块9: 文本解码器（生成最终报告）
        # 结合精炼的特征重新生成文本
        final_visual_features = self.feature_fusion(
            vit_features,
            refined_features
        )
        
        text_report = self.text_decoder(
            final_visual_features,
            text_embeddings
        )
        
        # ===== 输出层 =====
        # 文本报告和分割掩码
        output_text = self.output_projector.text_head(text_report)
        output_mask = self.output_projector.mask_head(refined_masks)
        
        return {
            'text_report': output_text,
            'segmentation_mask': output_mask,
            'expert_weights': expert_gates,
            'sam_prompts': sam_prompts,  # 可解释性
            'initial_masks': initial_masks,  # 对比展示
            'refined_masks': refined_masks
        }
    
    def decode_spatial_intent(self, spatial_intent):
        """
        将空间意图解码为SAM 3提示格式
        Args:
            spatial_intent: (B, D_prompt) LLM生成的空间意图
        Returns:
            dict: 包含点、框等提示
        """
        # 点提示生成头
        points = self.grounding_head.point_head(spatial_intent)  # (B, N_points, 2)
        points = torch.tanh(points)  # 归一化到[-1, 1]
        
        # 框提示生成头
        boxes = self.grounding_head.box_head(spatial_intent)  # (B, N_boxes, 4)
        boxes = torch.sigmoid(boxes)  # 归一化到[0, 1]
        
        # 可选: 掩码提示
        if hasattr(self.grounding_head, 'mask_head'):
            mask_prompts = self.grounding_head.mask_head(spatial_intent)
        else:
            mask_prompts = None
        
        return {
            'points': points,
            'boxes': boxes,
            'mask_prompts': mask_prompts
        }
