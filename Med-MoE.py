import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

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
# ========== Stage 1 模块 ==========

class VisualEncoder(nn.Module):
    """模块1: ViT编码器 + DoRA适配"""
    def __init__(self, config):
        super().__init__()
        # 预训练的ViT
        self.vit = AutoModel.from_pretrained(config.vit_model_name)
        
        # 冻结ViT主干
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # DoRA适配器（仅训练少量参数）
        self.dora_adapters = nn.ModuleList([
            DoRALayer(
                in_features=self.vit.config.hidden_size,
                rank=config.dora_rank
            ) for _ in range(self.vit.config.num_hidden_layers)
        ])
        
    def forward(self, image):
        # ViT前向
        outputs = self.vit(pixel_values=image, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # 所有层的输出
        
        # 应用DoRA适配器
        adapted_states = []
        for i, state in enumerate(hidden_states):
            if i < len(self.dora_adapters):
                adapted = self.dora_adapters[i](state)
                adapted_states.append(adapted)
            else:
                adapted_states.append(state)
        
        # 返回最后一层或所有层
        return adapted_states[-1]  # (B, N_patches+1, D)


class SAM3Engine(nn.Module):
    """模块2: SAM 3引擎（冻结）"""
    def __init__(self, config):
        super().__init__()
        # 加载预训练的SAM 3
        from segment_anything import sam_model_registry
        self.sam3 = sam_model_registry["vit_h"](checkpoint=config.sam3_checkpoint)
        
        # 完全冻结
        self.freeze_sam3()
        
        # 可选的轻量级Adapter（用于医学域适应）
        if config.use_sam_adapter:
            self.medical_adapter = SAM3MedicalAdapter(
                in_dim=256,  # SAM 3的特征维度
                adapter_type='parallel'
            )
        else:
            self.medical_adapter = None
    
    def freeze_sam3(self):
        for param in self.sam3.parameters():
            param.requires_grad = False
        self.sam3.eval()
    
    def forward(self, image, prompts=None, return_features=False):
        with torch.no_grad():  # 确保冻结
            # 图像编码
            image_embeddings = self.sam3.image_encoder(image)
            
            # 提示编码
            if prompts is not None:
                sparse_embeddings, dense_embeddings = self.sam3.prompt_encoder(
                    points=prompts.get('points'),
                    boxes=prompts.get('boxes'),
                    masks=prompts.get('mask_prompts')
                )
            else:
                # 无提示情况
                sparse_embeddings, dense_embeddings = None, None
            
            # 掩码解码
            masks, scores, low_res_masks = self.sam3.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.sam3.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True
            )
        
        # 可选的医学域适配
        if self.medical_adapter is not None:
            low_res_masks = self.medical_adapter(low_res_masks)
        
        output = {
            'masks': masks,  # (B, 3, 256, 256)
            'scores': scores,  # (B, 3)
            'features': low_res_masks  # (B, 256, 64, 64)
        }
        
        if return_features:
            output['image_embeddings'] = image_embeddings
        
        return output


class MultiModalFusion(nn.Module):
    """模块3: 特征融合 (ViT + SAM)"""
    def __init__(self, config):
        super().__init__()
        self.vit_proj = nn.Linear(config.vit_dim, config.fusion_dim)
        self.sam_proj = nn.Conv2d(config.sam_dim, config.fusion_dim, 1)
        
        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.fusion_dim,
            num_heads=config.num_attention_heads,
            batch_first=True
        )
        
        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(config.fusion_dim * 2, config.fusion_dim),
            nn.Sigmoid()
        )
    
    def forward(self, vit_features, sam_features):
        """
        vit_features: (B, N, D_vit)
        sam_features: (B, D_sam, H, W)
        """
        B, N, D_vit = vit_features.shape
        B, D_sam, H, W = sam_features.shape
        
        # 投影到统一空间
        vit_proj = self.vit_proj(vit_features)  # (B, N, D_fusion)
        
        # SAM特征需要从空间格式转换为序列
        sam_proj = self.sam_proj(sam_features)  # (B, D_fusion, H, W)
        sam_flat = sam_proj.flatten(2).transpose(1, 2)  # (B, H*W, D_fusion)
        
        # 跨模态注意力
        attended_vit, _ = self.cross_attention(
            query=vit_proj,
            key=sam_flat,
            value=sam_flat
        )
        
        # 门控融合
        gate_values = self.gate(
            torch.cat([vit_proj, attended_vit], dim=-1)
        )
        fused = gate_values * vit_proj + (1 - gate_values) * attended_vit
        
        return fused  # (B, N, D_fusion)


# ========== Stage 2 模块 ==========

class MOERouter(nn.Module):
    """模块4: MoE路由网络"""
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        
        # 路由网络
        self.router = nn.Sequential(
            nn.Linear(config.router_input_dim, config.router_hidden_dim),
            nn.LayerNorm(config.router_hidden_dim),
            nn.GELU(),
            nn.Linear(config.router_hidden_dim, self.num_experts),
            nn.Softmax(dim=-1)
        )
        
        # 负载均衡损失
        self.aux_loss_weight = config.aux_loss_weight
    
    def forward(self, x):
        """
        x: (B, D_input) 融合的视觉+文本特征
        Returns:
            expert_gates: (B, num_experts) 专家门控权重
        """
        return self.router(x)
    
    def compute_load_balance_loss(self, expert_gates):
        """计算负载均衡损失"""
        # 专家选择概率
        expert_probs = expert_gates.mean(dim=0)  # (num_experts,)
        
        # 理想均匀分布
        uniform_dist = torch.ones_like(expert_probs) / self.num_experts
        
        # KL散度作为损失
        load_balance_loss = F.kl_div(
            expert_probs.log(),
            uniform_dist,
            reduction='batchmean'
        )
        
        return load_balance_loss * self.aux_loss_weight


class ExpertPool(nn.Module):
    """模块5: 专家池 + 模块6: 专家聚合"""
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        
        # 定义不同类型的专家
        self.experts = nn.ModuleDict({
            'diagnosis': DiagnosisExpert(config),
            'description': DescriptionExpert(config),
            'quantification': QuantificationExpert(config),
            'guidance': GuidanceExpert(config),
            'localization': LocalizationExpert(config)
        })
        
        # 专家都使用DoRA
        for name, expert in self.experts.items():
            expert.apply_dora(config.dora_rank)
    
    def forward(self, visual_features, expert_gates, text_features=None):
        """
        visual_features: (B, N, D) 视觉特征
        expert_gates: (B, num_experts) 路由权重
        Returns:
            moe_output: (B, D) 专家聚合输出
        """
        B, N, D = visual_features.shape
        
        # 为每个样本选择top-k专家
        topk_weights, topk_indices = torch.topk(
            expert_gates, 
            k=min(2, self.num_experts),  # 选择top-2专家
            dim=-1
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # 初始化输出
        final_output = torch.zeros(B, D, device=visual_features.device)
        
        # 对每个专家进行处理
        for i in range(self.num_experts):
            # 找到需要当前专家的样本
            mask = (topk_indices == i).any(dim=-1)  # (B,)
            if not mask.any():
                continue
                
            # 获取样本权重
            sample_weights = torch.zeros(B, device=visual_features.device)
            for b in range(B):
                if mask[b]:
                    # 找到该专家在样本b中的位置
                    idx = (topk_indices[b] == i).nonzero(as_tuple=True)[0]
                    if idx.numel() > 0:
                        sample_weights[b] = topk_weights[b, idx[0]]
            
            # 提取需要处理的样本
            batch_indices = mask.nonzero(as_tuple=True)[0]
            batch_features = visual_features[batch_indices]
            
            # 专家前向
            expert_name = self.get_expert_name(i)
            expert_output = self.experts[expert_name](
                batch_features,
                text_features[batch_indices] if text_features is not None else None
            )
            
            # 加权聚合
            for j, b_idx in enumerate(batch_indices):
                final_output[b_idx] += sample_weights[b_idx] * expert_output[j]
        
        return final_output
    
    def get_expert_name(self, expert_idx):
        """映射专家索引到名称"""
        expert_names = list(self.experts.keys())
        return expert_names[expert_idx % len(expert_names)]


# ========== Stage 3 模块 ==========

class LLMCore(nn.Module):
    """模块7: LLM核心"""
    def __init__(self, config):
        super().__init__()
        # 使用轻量级LLM，如Phi-2, Qwen-7B等
        self.llm = AutoModel.from_pretrained(config.llm_model_name)
        
        # 冻结LLM的大部分层
        self.freeze_llm()
        
        # 仅微调最后几层或添加Adapter
        self.llm_adapter = LoRALayer(
            in_dim=self.llm.config.hidden_size,
            rank=config.lora_rank
        )
    
    def freeze_llm(self):
        # 冻结除最后2层外的所有参数
        total_layers = len(self.llm.encoder.layer)
        for i, layer in enumerate(self.llm.encoder.layer):
            if i < total_layers - 2:  # 保留最后2层可训练
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, input_embeds, text_embeddings):
        """
        input_embeds: (B, 1, D) MoE输出
        text_embeddings: (B, L, D_text) 文本指令
        """
        # 将MoE输出作为系统提示
        system_prompt = self.llm_adapter(input_embeds)
        
        # 构建LLM输入: [系统提示, 文本指令]
        llm_input = torch.cat([system_prompt, text_embeddings], dim=1)
        
        # LLM前向
        outputs = self.llm(
            inputs_embeds=llm_input,
            output_hidden_states=True
        )
        
        return outputs.last_hidden_state


class GroundingHead(nn.Module):
    """模块8: 接地模块（从LLM生成SAM提示）"""
    def __init__(self, config):
        super().__init__()
        # 空间意图提取
        self.intent_extractor = nn.Sequential(
            nn.Linear(config.llm_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, config.prompt_dim)
        )
        
        # 点提示生成
        self.point_head = nn.Sequential(
            nn.Linear(config.prompt_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 6),  # 3个点 * 2坐标
            nn.Tanh()  # 归一化到[-1, 1]
        )
        
        # 框提示生成
        self.box_head = nn.Sequential(
            nn.Linear(config.prompt_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 4),  # 1个框 * 4坐标
            nn.Sigmoid()  # 归一化到[0, 1]
        )
        
        # 掩码提示生成（可选）
        self.mask_head = nn.Sequential(
            nn.Linear(config.prompt_dim, 64*64),
            nn.Sigmoid()
        ) if config.use_mask_prompt else None
    
    def forward(self, llm_hidden_states):
        """
        llm_hidden_states: (B, L, D_llm)
        Returns:
            spatial_intent: (B, D_prompt)
        """
        # 使用CLS token或平均池化
        spatial_intent = llm_hidden_states[:, 0, :]  # CLS token
        # spatial_intent = llm_hidden_states.mean(dim=1)  # 或平均池化
        
        return self.intent_extractor(spatial_intent)


# ========== 输出层 ==========

class OutputProjector(nn.Module):
    """输出层投影"""
    def __init__(self, config):
        super().__init__()
        # 文本报告头
        self.text_head = nn.Sequential(
            nn.Linear(config.llm_dim, config.vocab_size),
            nn.LogSoftmax(dim=-1)
        )
        
        # 分割掩码头（精炼）
        self.mask_head = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 3个候选掩码
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, text_features, mask_features):
        text_output = self.text_head(text_features)
        mask_output = self.mask_head(mask_features)
        return text_output, mask_output

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
    

