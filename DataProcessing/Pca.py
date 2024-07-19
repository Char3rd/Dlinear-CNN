import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def pca(
    data: np.ndarray, n_components: int = 12, is_show_info: bool = False
) -> tuple[PCA, StandardScaler]:
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    pca = PCA(n_components=n_components)

    x = scaler.fit_transform(data)
    pca.fit(x)

    if is_show_info:
        show_info(pca)

    return (pca, scaler)


def show_info(pca: PCA) -> None:

    explained_variance_ratio = pca.explained_variance_ratio_
    num = len(explained_variance_ratio)
    component_names = [f"PC{i+1}" for i in range(num)]
    eigen_vectors = pca.components_
    eigen_values = pca.explained_variance_
    cumulative_variance_ratio = explained_variance_ratio.cumsum()

    print("特征向量:")
    print(eigen_vectors)
    print("特征值:")
    print(eigen_values)

    print("主成分贡献度:")
    for i, (label, variance_ratio) in enumerate(
        zip(component_names, explained_variance_ratio[:11]), start=1
    ):
        print(f"主成分 {label}: {variance_ratio:.4f}")

    # 绘制贡献度直方图
    plt.rcParams["font.sans-serif"] = ["Kaitt", "SimHei"]
    plt.figure(figsize=(10, 6))
    plt.bar(component_names, explained_variance_ratio)
    plt.xlabel("主成分")
    plt.ylabel("贡献度")
    plt.title("主成分贡献度")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    # 绘制贡献度的累计折线图
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(cumulative_variance_ratio) + 1),
        cumulative_variance_ratio,
        marker="o",
    )
    plt.xlabel("主成分数量")
    plt.ylabel("累计贡献度")
    plt.title("贡献度累计折线图")

    # 标注主成分的名称
    for i, txt in enumerate(component_names[: len(cumulative_variance_ratio)]):
        plt.annotate(txt, (i + 1, cumulative_variance_ratio[i]), fontsize=10)

    plt.grid(True)
    plt.show()

    # 绘制碎石图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(eigen_values) + 1), eigen_values, marker="o")

    plt.xlabel("主成分")
    plt.ylabel("特征根")
    plt.title("碎石图")

    # 标注主成分的名称
    for i, txt in enumerate(component_names[: len(eigen_values)]):
        plt.annotate(txt, (i + 1, eigen_values[i]), fontsize=10)

    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    input_file = r"D:\code\DLinear\data\train_FD001.csv"

    df = pd.read_csv(input_file)
    df["RL"] = ""
    # df.rename(columns={'时间':'date'}, inplace=True)
    columns_to_drop = df.columns[df.nunique() == 1].tolist()
    df.drop(columns_to_drop, axis=1, inplace=True)
    df.drop("P15", axis=1, inplace=True)
    df.drop("单元序号", axis=1, inplace=True)
    df.drop("时间", axis=1, inplace=True)

    print(df.values.shape)

    pca(data=df.values, is_show_info=True)

    pass
