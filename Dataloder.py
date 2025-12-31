class MedicalDatasetLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        
    def load_for_med_moe(self):
        """为Med-MoE加载多模态数据"""
        if self.dataset_name == "IU X-Ray":
            # 图像 + 报告 + 分割掩码（如果有）
            data = {
                "image": load_xray_image(),
                "text": ["描述胸片所见"],  # 或真实报告
                "segmentation": load_lung_mask(),  # 可选
                "bbox": load_finding_bboxes()  # 病变边界框
            }
        elif self.dataset_name == "VQA-RAD":
            data = {
                "image": load_radiology_image(),
                "question": "图像显示气胸吗？",
                "answer": "是的",
                "answer_type": "yes/no"
            }
        return data
