import argparse
import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    # 1. 解析 PAI AutoML 传入的超参数
    parser = argparse.ArgumentParser(description="Iris Classification for PAI AutoML")
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--num_leaves', type=int, default=31)
    parser.add_argument('--max_depth', type=int, default=-1)
    parser.add_argument('--feature_fraction', type=float, default=1.0)
    args = parser.parse_args()

    # 2. 自动加载内置数据集，无需读取 OSS
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. 初始化并训练 LightGBM 模型
    model = lgb.LGBMClassifier(
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        feature_fraction=args.feature_fraction,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train)

    # 4. 预测并计算准确率 (Accuracy)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 5. 输出指标供 PAI AutoML 收集
    # 注意：输出格式需根据 PAI 具体的正则解析规则调整，通常简单的 key: value 即可被识别
    print(f"accuracy={accuracy}")
    print(f"[PAI_METRIC] accuracy: {accuracy}")

if __name__ == "__main__":
    main()
