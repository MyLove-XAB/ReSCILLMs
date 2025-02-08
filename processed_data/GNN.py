import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
import settings
from os.path import join
from tqdm import tqdm
import matplotlib.pyplot as plt


# 定义TransE模型
class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin):
        super(TransE, self).__init__()
        self.margin = margin
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)

        # 初始化嵌入
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)

    def forward(self, pos_triples, neg_triples):
        # 获取头实体、关系、尾实体和负样本的嵌入
        head, relation, tail, neg_head, neg_tail = [], [], [], [], []
        for pos_triple, neg_triple in zip(pos_triples, neg_triples):
            head.append(pos_triple[0])
            relation.append(pos_triple[1])
            tail.append(pos_triple[2])
            neg_head.append(neg_triple[0])
            neg_tail.append(neg_triple[2])
        head, relation, tail, neg_head, neg_tail = torch.tensor(head, dtype=torch.long), torch.tensor(relation, dtype=torch.long), torch.tensor(tail, dtype=torch.long), torch.tensor(neg_head, dtype=torch.long), torch.tensor(neg_tail, dtype=torch.long)
        head_embedding = self.entity_embedding(head)
        relation_embedding = self.relation_embedding(relation)
        tail_embedding = self.entity_embedding(tail)
        negative_head_embedding = self.entity_embedding(neg_head)
        negative_tail_embedding = self.entity_embedding(neg_tail)

        # 计算正样本的距离（L2范数）
        positive_dist = torch.norm(head_embedding + relation_embedding - tail_embedding, p=2, dim=1)
        # 计算负样本的距离
        negative_dist = torch.norm(negative_head_embedding + relation_embedding - negative_tail_embedding, p=2, dim=1)

        # 损失函数是 margin-based ranking loss
        loss = torch.mean(torch.relu(positive_dist - negative_dist + self.margin))
        return loss

    def train_model(self, model, data_loader, optimizer, num_epochs):
        train_loader = data_loader[:int(len(data_loader) * 1)]
        # test_loader = data_loader[int(len(data_loader) * 0.8):]

        total_loss = []
        global_loss = 100
        total_test_loss = []
        for epoch in tqdm(range(num_epochs)):

            # Sbatch:list
            Sbatch = random.sample(train_loader, batch_size)
            Tbatch = []

            for triple in Sbatch:
                # 每个triple选3个负样例
                # for i in range(3):

                corrupted_triple = Corrupt(triple)
                # corrupted_triple = Corrupt(corrupted_triple)  # 替换头或尾实体，构造负样本
                if (triple, corrupted_triple) not in Tbatch:
                    Tbatch.append((triple, corrupted_triple))

            pos_triples = [p[0] for p in Tbatch]
            neg_triples = [p[1] for p in Tbatch]
            optimizer.zero_grad()
            loss = model(pos_triples, neg_triples)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

            # save the best model
            if global_loss > loss.item():
                global_loss = loss.item()
                save_model(model, join(save_dir, 'transE_model.pth'))

        print("final global loss: ", global_loss)

def Corrupt(triple):
    corrupted_triple = copy.deepcopy(triple)

    seed = random.random()

    if seed > 0.5:
        # 替换head
        rand_head = triple[0]
        while rand_head == triple[0]:
            rand_head = random.randint(0, num_entities - 1)
        corrupted_triple[0] = rand_head
    else:
        # 替换tail
        rand_tail = triple[2]
        while rand_tail == triple[2]:
            rand_tail = random.randint(0, num_entities - 1)
        corrupted_triple[2] = rand_tail
    return corrupted_triple


# 定义保存模型的函数
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)


# 定义保存实体和关系嵌入的函数
def save_embeddings(model, entity2id, relation2id, entity_embedding_file, relation_embedding_file):
    # 获取实体和关系的嵌入
    entity_embeddings = model.entity_embedding.weight.data.cpu().numpy()  # 实体嵌入矩阵
    relation_embeddings = model.relation_embedding.weight.data.cpu().numpy()  # 关系嵌入矩阵

    # 将实体嵌入保存为字典
    entity_embedding_dict = {entity: entity_embeddings[idx].tolist() for entity, idx in entity2id.items()}
    # 将关系嵌入保存为字典
    relation_embedding_dict = {relation: relation_embeddings[idx].tolist() for relation, idx in relation2id.items()}

    # 保存实体嵌入字典
    torch.save(entity_embedding_dict, entity_embedding_file)
    # 保存关系嵌入字典
    torch.save(relation_embedding_dict, relation_embedding_file)


def load_embeddings():
    # 读取实体和关系嵌入
    entity_embeddings = torch.load(join(save_dir, 'TransE_ett_emb.pth'))
    relation_embeddings = torch.load(join(save_dir, 'TransE_relation_emb.pth'))

    # 查看某个实体或关系的嵌入向量
    print(entity_embeddings['5db80dc83a55acd5c14a24b9b5'])  # 输出实体1的嵌入向量
    print(relation_embeddings['cites'])  # 输出关系1的嵌入向量

    # 计算实体之间的相似度
    entity1 = '5db80dc83a55acd5c14a24b9'
    entity2 = '5db80dc83a55acd5c14a24b9b5'

    entity1_emb = torch.tensor(entity_embeddings[entity1])
    entity2_emb = torch.tensor(entity_embeddings[entity2])
    similarity = torch.cosine_similarity(entity1_emb, entity2_emb, dim=0)
    print(f"实体{entity1}和实体{entity2}的相似度为: {similarity.item()}")


save_dir = join(settings.PROJ_DIR, "data_processed")
with open(join(save_dir, "ett2intID.txt"), 'r', encoding="utf-8") as f:
    ett2intID = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in f.readlines()}
with open(join(save_dir, "strID2intID.txt"), 'r') as f:
    strID2intID = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in f.readlines()}
with open(join(save_dir, "relation2intID.txt"), 'r') as f:
    relation2intID = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in f.readlines()}

num_entities = len(ett2intID) + len(strID2intID)      # 实体数量
num_relations = len(relation2intID)  # 关系数量
batch_size = 64  # batch大小


# 示例：模型训练与保存
def main():
    # 模型参数

    embedding_dim = 200  # 嵌入维度
    margin = 1.0  # margin值, 用于计算margin-based ranking loss
    num_epochs = 500  # 训练轮次
    learning_rate = 1e-3

    # 创建TransE模型
    model = TransE(num_entities, num_relations, embedding_dim, margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    with open(join(save_dir, "knowledge_graph_triples_ids.txt"), 'r', encoding='utf-8') as file:
        data = [[int(line.strip().split('\t')[0]), int(line.strip().split("\t")[1]), int(line.strip().split("\t")[2])] for line in
                file.readlines()]

    #  shuffle
    random.shuffle(data)

    # data_loader = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
    model.train()
    # 训练模型
    model.train_model(model, data, optimizer, num_epochs)

    # strID和ettID共同组成ett
    ett = {**strID2intID, **ett2intID}
    # load saved best model
    model = TransE(num_entities, num_relations, embedding_dim, margin)
    model.load_state_dict(torch.load(join(save_dir, 'transE_model.pth')))

    # 保存实体和关系的嵌入向量
    save_embeddings(model, ett, relation2intID, join(save_dir, 'TransE_ett_emb.pth'), join(save_dir, 'TransE_relation_emb.pth'))

    load_embeddings()


# 主程序入口
if __name__ == "__main__":
    main()


