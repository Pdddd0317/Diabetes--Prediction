"""
训练与评估模型的脚本。
- 仅使用 sklearn 自带的 load_diabetes 数据集
- 训练线性回归与随机森林回归模型
- 计算并打印 MSE、R²
- 保存模型、评估指标、可视化图像，供网页应用读取
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# 定义输出目录，Path 对象便于跨平台路径处理
MODELS_DIR = Path("models")
FIGURES_DIR = Path("figures")
METRICS_PATH = MODELS_DIR / "metrics.json"


def ensure_dirs() -> None:
    """创建保存模型与图像的目录。"""
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)


def load_data():
    """加载糖尿病数据集并返回特征、标签及特征名。"""
    data = load_diabetes()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    return X, y, feature_names


def train_and_evaluate(X: np.ndarray, y: np.ndarray, feature_names: list[str]):
    """划分数据集、训练模型并返回评估结果与训练好的模型。"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 训练线性回归模型
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # 训练随机森林回归模型
    rf_model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # 在测试集上预测
    lr_preds = lr_model.predict(X_test)
    rf_preds = rf_model.predict(X_test)

    # 计算指标
    metrics = {
        "linear_regression": {
            "mse": mean_squared_error(y_test, lr_preds),
            "r2": r2_score(y_test, lr_preds),
        },
        "random_forest": {
            "mse": mean_squared_error(y_test, rf_preds),
            "r2": r2_score(y_test, rf_preds),
        },
        "feature_names": feature_names,
    }

    print("特征名称:", feature_names)
    print(f"数据维度: 样本数={X.shape[0]}, 特征数={X.shape[1]}")
    print("线性回归 -> MSE: {:.4f}, R²: {:.4f}".format(
        metrics["linear_regression"]["mse"], metrics["linear_regression"]["r2"]
    ))
    print("随机森林 -> MSE: {:.4f}, R²: {:.4f}".format(
        metrics["random_forest"]["mse"], metrics["random_forest"]["r2"]
    ))

    return (X_train, X_test, y_train, y_test), (lr_model, rf_model), metrics, rf_preds


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Path:
    """绘制真实值 vs 预测值散点图并保存。"""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, color="tab:blue", edgecolors="white")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.title("随机森林预测 vs 真实值")
    plt.tight_layout()
    output_path = FIGURES_DIR / "rf_true_vs_pred.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_feature_importance(model: RandomForestRegressor, feature_names: list[str]) -> Path:
    """绘制随机森林的特征重要性条形图并保存。"""
    importances = model.feature_importances_
    # 根据重要性排序，便于观察
    indices = np.argsort(importances)[::-1]
    sorted_features = np.array(feature_names)[indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(8, 6))
    plt.bar(range(len(sorted_importances)), sorted_importances, color="tab:green")
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha="right")
    plt.ylabel("特征重要性")
    plt.title("随机森林特征重要性")
    plt.tight_layout()
    output_path = FIGURES_DIR / "rf_feature_importance.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def save_models(lr_model: LinearRegression, rf_model: RandomForestRegressor) -> None:
    """保存训练好的模型以供后续加载。"""
    dump(lr_model, MODELS_DIR / "linear_regression_model.joblib")
    dump(rf_model, MODELS_DIR / "random_forest_model.joblib")


def save_metrics(metrics: dict) -> None:
    """将评估指标保存为 JSON，方便网页读取。"""
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main():
    ensure_dirs()
    X, y, feature_names = load_data()
    dataset, models, metrics, rf_preds = train_and_evaluate(X, y, feature_names)

    # dataset 包含划分的数据，取测试集真实值用于绘图
    _, X_test, _, y_test = dataset
    _, rf_model = models

    # 绘制并保存图像
    plot_predictions(y_test, rf_preds)
    plot_feature_importance(rf_model, feature_names)

    # 保存模型和指标
    save_models(models[0], models[1])
    save_metrics(metrics)

    print("训练与保存完成，模型与图像已存储在 'models' 与 'figures' 目录下。")


if __name__ == "__main__":
    main()
