# 糖尿病指标预测小项目

基于 sklearn 自带的 `load_diabetes` 数据集，包含模型训练与 Streamlit 网页展示，适合初学者学习端到端流程。网页仅用于学习演示，不能替代专业医疗诊断。

## 文件结构
- `requirements.txt`：项目依赖列表。
- `train_model.py`：训练线性回归与随机森林模型、评估并保存可视化。
- `app.py`：Streamlit 网页应用，支持在线输入特征并预测。
- `models/`：存放训练好的模型（运行训练脚本后生成）。
- `figures/`：存放训练可视化图像（运行训练脚本后生成）。
- `models/metrics.json`：保存的测试集指标（MSE、R² 和特征名）。

## 环境准备（任选其一）
- **Python venv**
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Windows 使用 .venv\\Scripts\\activate
  pip install -r requirements.txt
  ```
- **conda**
  ```bash
  conda create -n diabetes-pred python=3.10 -y
  conda activate diabetes-pred
  pip install -r requirements.txt
  ```

## 运行流程（建议按顺序执行）
1. **激活虚拟环境**：确保使用上面创建的 venv 或 conda 环境。
2. **训练模型并生成文件**：
   ```bash
   python train_model.py
   ```
   运行后会在 `models/` 中得到 `linear_regression_model.joblib`、`random_forest_model.joblib`、`metrics.json`，在 `figures/` 中得到 `rf_true_vs_pred.png`、`rf_feature_importance.png`。
3. **启动网页应用**：
   ```bash
   streamlit run app.py
   ```
   浏览器会打开本地地址（通常是 http://localhost:8501），界面包含模型指标、特征输入表单、模型选择、预测按钮以及两张可视化图。

## 网页功能说明
- **模型指标展示**：从 `models/metrics.json` 读取线性回归与随机森林在测试集上的 MSE、R²，便于对比模型表现。
- **特征输入表单**：对 `load_diabetes` 的每个特征提供数字输入框（默认 0.0），支持浮点数。输入完成后选择模型并点击“开始预测”即可得到预测值。
- **可视化**：
  - `rf_true_vs_pred.png`：展示随机森林在测试集上的真实值与预测值散点图，点越贴近对角线表示预测越准确。
  - `rf_feature_importance.png`：随机森林的特征重要性条形图，数值越大表示该特征对模型越重要。
- **缺失文件提示**：如模型或图片不存在，界面会提示先运行 `python train_model.py`。

## 指标含义（医学小白也能理解）
- **MSE（均方误差）**：衡量预测值与真实值的平均平方差，值越小越好；常见单位是目标的平方单位。
- **R²（决定系数）**：衡量模型对数据的解释程度，范围通常在 0 到 1，越接近 1 说明模型越能解释数据（若出现负值表示模型效果比简单的平均值预测还差）。

## 常见问题
- **启动网页找不到模型或图片**：先运行 `python train_model.py` 生成需要的文件，再执行 `streamlit run app.py`。
- **依赖安装失败**：检查网络或更换镜像源，确认已激活虚拟环境后再安装。

> 本项目仅供学习与演示，不提供任何医疗建议。
