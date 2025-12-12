# 糖尿病指标预测小项目

基于 sklearn 自带的 `load_diabetes` 数据集，包含模型训练与 Streamlit 网页展示，适合初学者学习端到端流程。

## 文件结构
- `requirements.txt`：项目依赖列表。
- `train_model.py`：训练线性回归与随机森林模型、评估并保存可视化。
- `app.py`：Streamlit 网页应用，支持在线输入特征并预测。
- `models/`：存放训练好的模型（运行训练脚本后生成）。
- `figures/`：存放训练可视化图像（运行训练脚本后生成）。
- `metrics.json`：保存的测试集指标。

## 快速开始
1. 创建虚拟环境并安装依赖：
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows 使用 .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```
2. 训练模型并生成图像：
   ```bash
   python train_model.py
   ```
3. 启动网页应用（需保持虚拟环境已激活）：
   ```bash
   streamlit run app.py
   ```

> 提示：模型与图像由训练脚本自动生成，网页启动前请先运行训练脚本。网页仅供学习演示，不可替代医疗诊断。
