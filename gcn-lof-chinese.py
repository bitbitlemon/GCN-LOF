from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
from utils import load_data
from torch_geometric.utils import from_scipy_sparse_matrix

# 设置随机种子和基本参数
seed = 42
DATA_PATH = "./full_datasets/"
BATCH_SIZE = 128

# 定义函数将数据转换为图数据
def create_graph_data(x, y, k=10):
    from sklearn.neighbors import kneighbors_graph

    # 使用k近邻计算邻接矩阵
    adj = kneighbors_graph(x, n_neighbors=k, mode='connectivity', include_self=False)
    edge_index, _ = from_scipy_sparse_matrix(adj)

    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y.values, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data

# 加载数据集并转换为图数据
cids = []
for _, _, cid in os.walk(DATA_PATH):
    cids.extend(cid)

silos = {}

for cid in cids:
    _cid = cid[:cid.find(".csv")]
    silos[_cid] = {}
    x_train, y_train, x_test, y_test = load_data.load_data(os.path.join(DATA_PATH, cid), info=False)

    # 转换为图数据
    train_data = create_graph_data(x_train, y_train)
    test_data = create_graph_data(x_test, y_test)

    silos[_cid]["train_data"] = train_data
    silos[_cid]["test_data"] = test_data

# 定义GCN模型
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=16):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 训练和测试GCN模型，增加过拟合检测
num_epochs = 50
learning_rate = 0.01

gcn_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

for silo_name, silo_data in silos.items():
    print(f"> Training GCN on Silo: {silo_name}")

    train_data = silo_data["train_data"]
    test_data = silo_data["test_data"]

    num_features = train_data.num_features
    num_classes = len(torch.unique(train_data.y))

    model = GCN(num_features=num_features, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    # 记录训练和测试损失
    train_losses = []
    test_losses = []

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(train_data)
        train_loss = F.nll_loss(out, train_data.y)
        train_loss.backward()
        optimizer.step()

        # 记录训练损失
        train_losses.append(train_loss.item())

        # 在测试集上评估
        model.eval()
        with torch.no_grad():
            out_test = model(test_data)
            test_loss = F.nll_loss(out_test, test_data.y)
            test_losses.append(test_loss.item())

        model.train()

    # 保存训练和测试损失到CSV文件
    loss_data = pd.DataFrame(
        {"Epoch": range(1, num_epochs + 1), "Training Loss": train_losses, "Testing Loss": test_losses})
    loss_data.to_csv(f"{silo_name}_loss_data.csv", index=False)

    # 在测试集上评估模型性能
    model.eval()
    out = model(test_data)
    pred = out.argmax(dim=1).detach().numpy()
    y_true = test_data.y.detach().numpy()

    acc = accuracy_score(y_true, pred)
    pre = precision_score(y_true, pred, pos_label=1)
    rec = recall_score(y_true, pred, pos_label=1)
    f1 = f1_score(y_true, pred, pos_label=1)

    gcn_metrics["accuracy"].append(acc)
    gcn_metrics["precision"].append(pre)
    gcn_metrics["recall"].append(rec)
    gcn_metrics["f1"].append(f1)

    print(f">> GCN Metrics on {silo_name}: ACC={acc:.4f}, PRE={pre:.4f}, REC={rec:.4f}, F1={f1:.4f}")

# 输出平均结果
print(f">> Average GCN Accuracy: {np.mean(gcn_metrics['accuracy']):.4f} ± {np.std(gcn_metrics['accuracy']):.4f}")
print(f">> Average GCN Precision: {np.mean(gcn_metrics['precision']):.4f} ± {np.std(gcn_metrics['precision']):.4f}")
print(f">> Average GCN Recall: {np.mean(gcn_metrics['recall']):.4f} ± {np.std(gcn_metrics['recall']):.4f}")
print(f">> Average GCN F1-score: {np.mean(gcn_metrics['f1']):.4f} ± {np.std(gcn_metrics['f1']):.4f}")
