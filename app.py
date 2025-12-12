"""
简易 Streamlit 网页应用：
- 展示模型评估指标
- 提供特征输入与模型选择进行预测
- 展示训练时保存的可视化图像
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.datasets import load_diabetes

# 常量路径定义
MODELS_DIR = Path("models")
FIGURES_DIR = Path("figures")
METRICS_PATH = MODELS_DIR / "metrics.json"


@st.cache_resource(show_spinner=False)
def load_models():
    """加载已训练好的线性回归与随机森林模型。"""
    lr_path = MODELS_DIR / "linear_regression.joblib"
    rf_path = MODELS_DIR / "random_forest.joblib"
    if not lr_path.exists() or not rf_path.exists():
        st.error("未找到模型文件，请先运行 train_model.py 生成模型。")
        st.stop()
    lr_model = load(lr_path)
    rf_model = load(rf_path)
    return {"线性回归": lr_model, "随机森林回归": rf_model}


@st.cache_data(show_spinner=False)
def load_metrics() -> Dict:
    """读取评估指标 JSON。"""
    if not METRICS_PATH.exists():
        st.error("未找到 metrics.json，请先运行 train_model.py。")
        st.stop()
    with METRICS_PATH.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics


def load_dataset_stats():
    """加载数据集以获得特征名称和均值，方便给默认输入值。"""
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    return data.feature_names, df.mean().to_dict()


def build_feature_inputs(feature_names: List[str], default_values: Dict[str, float]):
    """在界面上生成特征输入控件。"""
    user_inputs = []
    cols = st.columns(2)
    for idx, feature in enumerate(feature_names):
        col = cols[idx % 2]
        # 使用 number_input 强制为浮点数，并提供默认值
        value = col.number_input(
            label=f"{feature} (浮点数)",
            value=float(round(default_values.get(feature, 0.0), 3)),
            step=0.01,
            format="%.4f",
        )
        user_inputs.append(value)
    return user_inputs


def predict(model, features: List[float]):
    """使用指定模型进行预测，返回预测值。"""
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    return float(prediction[0])


def main():
    st.set_page_config(page_title="糖尿病指标预测演示", page_icon="🩺", layout="wide")
    st.title("🩺 糖尿病指标预测小工具")
    st.markdown(
        """
        本项目基于 sklearn 自带的 **load_diabetes** 数据集，仅用于学习演示。
        你可以在左侧输入各项特征值，选择模型后点击“预测”查看结果。
        """
    )

    # 侧边栏：展示指标
    st.sidebar.header("模型表现（测试集）")
    metrics = load_metrics()
    lr_metrics = metrics.get("linear_regression", {})
    rf_metrics = metrics.get("random_forest", {})
    st.sidebar.metric("线性回归 MSE", f"{lr_metrics.get('mse', 0):.2f}")
    st.sidebar.metric("线性回归 R²", f"{lr_metrics.get('r2', 0):.2f}")
    st.sidebar.metric("随机森林 MSE", f"{rf_metrics.get('mse', 0):.2f}")
    st.sidebar.metric("随机森林 R²", f"{rf_metrics.get('r2', 0):.2f}")

    # 主区域布局
    left, right = st.columns([1.1, 0.9])

    with left:
        st.subheader("输入特征并进行预测")
        feature_names, defaults = load_dataset_stats()
        user_inputs = build_feature_inputs(feature_names, defaults)

        model_options = ["线性回归", "随机森林回归"]
        selected_model_name = st.selectbox("选择预测模型", model_options)

        models = load_models()
        model = models[selected_model_name]

        if st.button("🚀 预测", type="primary"):
            try:
                pred = predict(model, user_inputs)
                st.success(f"预测结果：{pred:.2f}")
            except Exception as exc:  # 捕获异常并友好提示
                st.error(f"预测失败，请检查输入是否为数字。错误信息：{exc}")

    with right:
        st.subheader("训练阶段的可视化")
        pred_img = FIGURES_DIR / "rf_true_vs_pred.png"
        imp_img = FIGURES_DIR / "rf_feature_importance.png"

        if pred_img.exists():
            st.markdown("**随机森林：真实值 vs 预测值**")
            st.image(str(pred_img))
        else:
            st.info("未找到预测散点图，请先运行 train_model.py 生成。")

        if imp_img.exists():
            st.markdown("**随机森林：特征重要性**")
            st.image(str(imp_img))
        else:
            st.info("未找到特征重要性图，请先运行 train_model.py 生成。")

    st.markdown("---")
    st.markdown(
        "本工具仅供学习与演示，不能替代医生诊断。数据来源：sklearn `load_diabetes`。"
    )


if __name__ == "__main__":
    main()
