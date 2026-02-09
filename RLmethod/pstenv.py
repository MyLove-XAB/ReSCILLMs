import random

import numpy as np
import torch
import settings
import utils
from os.path import join
from transformers import BertForSequenceClassification, BertTokenizer, LlamaForSequenceClassification, LlamaTokenizer
from tqdm import tqdm
import json
from collections import defaultdict as dd
from sklearn.metrics import average_precision_score


class PSTEnv(object):
    def __init__(self, text=True, graph=True, ranking=False):
        self.previous_paper_emb = None
        self.text = text        # whether to use text information
        self.graph = graph      # whether to use graph embedding
        # train and valid data
        self.train_data = utils.load_json(save_dir, "Train_papers_refs_pair.json")
        self.valid_data = utils.load_json(save_dir, "Valid_papers_refs_pair.json")
        self.gt = utils.load_json(valid_dir, "ground_truths_valid.json")

        # train and valid keys determine the data order
        self.train_keys = list(self.train_data.keys())
        # shuffle
        random.shuffle(self.train_keys)
        self.valid_keys = list(self.valid_data.keys())

        # train and valid ground truth
        self.truth_label = utils.load_json(save_dir, "ground_truth_label.json")
        self.truth_index = utils.load_json(save_dir, "ground_truth_index.json")
        self.truth_bid = utils.load_json(save_dir, "ground_truth_bid.json")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.entity_embeddings = {k: torch.tensor(v, dtype=torch.float32, device=self.device)
                                  for k, v in torch.load(join(save_dir, 'TransE_ett_emb.pth')).items()}
        self.default_emb = torch.zeros(200, dtype=torch.bfloat16, device=self.device)  # 根据 feature_size 调整
        # self.entity_embeddings = torch.load(join(save_dir, 'TransE_ett_emb.pth'))

        # 【优化 2】预解析所有 GT ID 映射，避免 reset 时重复运行那 15 行判断逻辑
        self.gt_id_map = {}
        for qp in (self.train_keys + self.valid_keys):
            temp_gt = []
            for i, tmp in enumerate(self.truth_index.get(qp, [])):
                if tmp and tmp in self.entity_embeddings:
                    temp_gt.append(tmp)
                else:
                    try:
                        gt = qp + self.truth_bid[qp][i]
                        if gt in self.entity_embeddings:
                            temp_gt.append(gt)
                    except:
                        continue
            self.gt_id_map[qp] = temp_gt if temp_gt else [self.query_paper]

        self.gt_matrix_map = {}
        for qp, ids in self.gt_id_map.items():
            # 将该 paper 对应的所有 GT embedding 堆叠成 [N, Dim] 的矩阵
            self.gt_matrix_map[qp] = torch.stack([self.entity_embeddings.get(i, self.default_emb) for i in ids])

        self.index = -1
        # self.batch_size = 32
        self.ranking = ranking      # whether to use ranking reward
        self.test_iter = 0
        self.count = 0

    def reset(self):
        # get the initial state
        self.found = False      # record if the ground truth is found
        self.score = 0.5
        self.info = 0          # save map for finally
        self.finally_found = False

        self.steps = 0
        # self.index = random.randint(0, len(self.train_keys))
        self.index += 1          # random.randint(0, len(self.train_keys)), or sequentially
        self.done = False
        self.query_paper = self.train_keys[self.index % len(self.train_keys)]           # mod to avoid out of index
        self.cur_data = self.train_data[self.query_paper]

        self.previous_paper = self.cur_data[self.steps]         # used to save the best paper
        self.cur_paper = self.cur_data[self.steps + 1]

        # 直接从缓存取 GT IDs，并堆叠为矩阵, reset 阶段预先堆叠：
        # self.gt_ls = self.gt_id_map[self.query_paper]
        self.global_paper_matrix = self.gt_matrix_map[self.query_paper]

        p_id = self.previous_paper[0] or (self.query_paper + self.previous_paper[1])
        self.previous_paper_emb = self.entity_embeddings.get(p_id, self.default_emb)

        # 计算阶段一次性广播：
        # cosine_similarity 支持 [num_gt, dim] 与 [dim] 的广播计算
        sims = torch.nn.functional.cosine_similarity(self.global_paper_matrix, self.previous_paper_emb.unsqueeze(0))
        self.previous_similarity = sims.max()
        # self.previous_similarity = max([torch.cosine_similarity(global_paper_emb, self.previous_paper_emb, dim=0) for global_paper_emb in self.global_paper_emb_ls])

        self.global_state = self.query_paper

        self.state = [self.previous_paper[1], self.cur_paper[1]]            # ["bid", "bid"]

        # return the initial state
        return self.state, self.global_state

    def ranking_reward(self):
        reward = 0
        if self.done:
            # calculate the ranking reward
            self.info = np.array(self.info)
            rank = np.argsort(self.info)
            for pos in range(min(10, len(rank))):
                # if self.train_truth_label[rank[pos]] == 1:
                if self.truth_label[rank[pos]] == 1:
                    reward = 1/(pos + 1)
        # 如果依照概率选择动作，最后的top1可能并不是最后保留下来的
        return reward

    def naive_reward(self):
        # just see if the final result is the same as the ground truth
        if self.done:
            # if self.cur_paper[0] in self.train_truth_index[self.query_paper]:
            if self.truth_bid[self.query_paper]:       # 如果有ground truth bid
                if self.previous_paper[1] in self.truth_bid[self.query_paper]:
                    reward = 1
                    self.finally_found = True
                else:
                    reward = -1         # 最终没找到，reward为负
                    # self.finally_found = False
            else:           # reward should be zero, because we don't know whether the bid is correct
                if self.previous_paper[0] in self.truth_index[self.query_paper]:
                    reward = 1
                    self.finally_found = True
                else:
                    reward = -1         # 没有truth_bid，reward应该为0， -5
                    # self.finally_found = False
        else:
            if self.truth_bid[self.query_paper]:       # 如果有ground truth bid
                if self.previous_paper[1] in self.truth_bid[self.query_paper]:
                    reward = 1
                    self.found = True
                else:
                    if self.found:      # 如果前一个已经找到了正确的，那么后一个是负的
                        reward = -1      # / self.steps
                        self.found = False      # reset found flag
                    else:
                        reward = 0
            else:           # no bid, we can only use the truth_index
                if self.previous_paper[0] in self.truth_index[self.query_paper]:
                    reward = 1
                    self.found = True
                else:
                    if self.found:
                        reward = -1
                        self.found = False
                    else:
                        reward = 0
            # reward = 0
        return reward

    def get_reward(self):
        # if ranking, return ranking_reward()
        if self.ranking:
            return self.ranking_reward()            # 并没有用到
        # else, return naive_reward()
        else:
            return self.naive_reward()

    def calculate_score(self):
        pass

    def step(self, action):
        self.steps += 1
        # get the reward

        # if action > 0.5:        # deterministic
        if np.random.random() < action:     # stochastic, 认为cur_paper优于previous_paper, 则更新pre为cur
            cur_id = self.cur_paper[0] or (self.query_paper + self.cur_paper[1])
            self.cur_paper_emb = self.entity_embeddings.get(cur_id, self.default_emb)
            cur_sims = torch.cosine_similarity(self.global_paper_matrix, self.cur_paper_emb.unsqueeze(0))
            self.cur_similarity = cur_sims.max()

            graph_reward = (self.cur_similarity - self.previous_similarity).item()
            self.previous_paper = self.cur_paper
            self.previous_similarity = self.cur_similarity       # update
        else:
            graph_reward = 0        # 如果不更新previous_paper，graph_reward为0

        # get the done flag
        if self.steps >= len(self.cur_data) - 1:
            self.done = True

        else:
            self.done = False
            # get the next state
            # 否则previous_paper仍然是当前最好的，不更新previous_paper，只更新cur_paper
            self.cur_paper = self.cur_data[self.steps + 1]
            self.state = [self.previous_paper[1], self.cur_paper[1]]

        # graph_reward = self.previous_similarity

        return self.state, self.get_reward() + graph_reward, self.done, self.info, self.steps          # self.get_reward()+graph_reward

    def test_env(self, actor):
        self.case_result = dict()  # used to save the successful cases
        self.test_index = -1
        map_ls = []
        self.success = 0
        # self.reward_dic = dd(list)
        self.info_dic = dd(list)
        self.prob = dd(list)
        self.test_iter += 1

        for i in range(len(self.valid_data)):        # 每次只测试一半的数据
            # get the initial state
            self.found = False
            self.score = 0.5
            self.info = [self.score]  # save score for each step
            self.finally_found = False
            self.steps = 0

            self.test_index += 1  # random.randint(0, len(self.train_keys)), or sequentially
            self.done = False
            self.query_paper = self.valid_keys[self.test_index % len(self.valid_data)]  # mod to avoid out of index
            self.cur_data = self.valid_data[self.query_paper]

            self.previous_paper = self.cur_data[self.steps]
            self.cur_paper = self.cur_data[self.steps + 1]

            self.global_paper_matrix = self.gt_matrix_map[self.query_paper]

            p_id = self.previous_paper[0] or (self.query_paper + self.previous_paper[1])
            self.previous_paper_emb = self.entity_embeddings.get(p_id, self.default_emb)
            self.previous_similarity = torch.nn.functional.cosine_similarity(self.global_paper_matrix, self.previous_paper_emb.unsqueeze(0)).max()

            self.global_state = self.query_paper
            actor.global_paper = self.global_state

            self.state = [self.previous_paper[1], self.cur_paper[1]]        # ["bid", "bid"]
            done = False
            rewards = []
            p_list = []
            # state = self.state

            actor.get_global_embedding(self.global_state, signal="test")
            self.prev_ls = list()
            self.prev_ls.append(self.previous_paper)

            while not done:
                # p = [0, 0]
                action, log_prob, p, v = actor.get_action(self.state, signal="test")          # use cls token
                p_list.append(float(p[1]))
                action = 1 if p[1] > 0.5 else 0
                next_state, reward, done, info_, step_ = self.step(action)       # p[1]: probability of action==1
                # log_probs.append(log_prob)
                rewards.append(reward)          # .item()
                self.state = next_state
                self.prev_ls.append(self.previous_paper)

            predict = np.zeros(len(self.cur_data))
            for i, prev in enumerate(self.prev_ls):
                if predict[int(prev[1][1:])] == 0:
                    predict[int(prev[1][1:])] = (i + 1) / len(self.prev_ls)
            tmp_map = average_precision_score(self.gt[self.query_paper], predict)

            map_ls.append(tmp_map)
            if self.finally_found:            # if finally found, then success
                self.success += 1
        return self.success / len(self.valid_data), sum(map_ls)/len(map_ls)      # success rate

    def case(self, actor):
        self.case_result = dict()        # used to save the successful cases
        self.test_index = -1
        map_ls = []
        self.success = 0
        # self.reward_dic = dd(list)
        self.info_dic = dd(list)
        self.prob = dd(list)
        self.test_iter += 1

        for i in tqdm(range(len(self.valid_data))):  # 每次只测试一半的数据
            # get the initial state
            self.found = False
            self.score = 0.5
            self.info = [self.score]  # save score for each step
            self.finally_found = False
            self.steps = 0

            self.test_index += 1  # random.randint(0, len(self.train_keys)), or sequentially
            self.done = False
            self.query_paper = self.valid_keys[self.test_index % len(self.valid_data)]  # mod to avoid out of index
            self.cur_data = self.valid_data[self.query_paper]

            self.previous_paper = self.cur_data[self.steps]
            self.cur_paper = self.cur_data[self.steps + 1]

            self.global_paper_matrix = self.global_paper_matrix[self.query_paper]

            p_id = self.previous_paper[0] or (self.query_paper + self.previous_paper[1])
            self.previous_paper_emb = self.entity_embeddings.get(p_id, self.default_emb)
            self.previous_similarity = torch.nn.functional.cosine_similarity(self.global_paper_matrix, self.previous_paper_emb.unsqueeze(0)).max()

            self.global_state = self.query_paper
            actor.global_paper = self.global_state

            self.state = [self.previous_paper[1], self.cur_paper[1]]  # ["bid", "bid"]
            done = False
            rewards = []
            p_list = []
            # state = self.state

            actor.get_global_embedding(self.global_state, signal="test")
            self.prev_ls = list()
            self.prev_ls.append(self.previous_paper)

            while not done:
                # p = [0, 0]
                action, log_prob, p, v = actor.get_action(self.state, signal="test")  # use cls token
                p_list.append(float(p[1]))
                action = 1 if p[1] > 0.5 else 0
                next_state, reward, done, info_, step_ = self.step(action)  # p[1]: probability of action==1

                self.state = next_state
                self.prev_ls.append(self.previous_paper)

            # write map result
            predict = np.zeros(len(self.cur_data))
            for i, prev in enumerate(self.prev_ls):
                if predict[int(prev[1][1:])] == 0:
                    predict[int(prev[1][1:])] = (i+1)/len(self.prev_ls)

            tmp_map = average_precision_score(self.gt[self.query_paper], predict)
            if self.finally_found:
                self.case_result[self.query_paper] = ("found", self.prev_ls)
            else:
                self.case_result[self.query_paper] = ("not found", self.prev_ls)

            map_ls.append(tmp_map)
            if self.finally_found:  # if finally found, then success
                self.success += 1

        return self.success / len(self.valid_data), sum(map_ls) / len(map_ls)  # success rate


def extract_paper_text_info(paper_id, query_paper, content=False, context_pos=False):
    # get the paper text information, including title, abstract, keywords, context
    text_ls = []
    if paper_id in paper_info.keys():
        text_ls.append(paper_info[paper_id]["title"])
        text_ls.append(paper_info[paper_id]["abstract"])
        text_ls.append("[SEP]".join(paper_info[paper_id]["keywords"]))
        text_ls.append("[SEP]".join(paper_info[paper_id]["authors"]))
        text_ls.append(paper_info[paper_id]["venue"])
        if content:
            # only query paper has the content
            text_ls.append(paper_info[paper_id]["content"])

    try:
        if context_pos:         # query paper does not have context information
            for i in range(6-len(text_ls)):
                text_ls.append("[SEP]")         # 不存在的用[SEP]占位
            context = bib_contexts[query_paper][context_pos]
            text_ls.append(context)
    except:
        print(f"paper {paper_id} does not have context information")
    return text_ls


data_dir = join(settings.DATA_TRACE_DIR, "PST")
save_dir = join(settings.PROJ_DIR, "data_processed")
valid_dir = join(settings.DATA_TRACE_DIR, "PST-valid")
in_dir = join(data_dir, "paper-xml")
dblp_fname = "DBLP-Citation-network-V15.1.json"

bib_contexts = utils.load_json(save_dir, "bib_to_contexts.json")

scibert = join(settings.PROJ_DIR, "pretrain_models/bertmodel")
llama2 = join(settings.PROJ_DIR, "pretrain_models/llmamodel/llama2-7b-hf")
paper_info = utils.load_json(save_dir, "paper_info.json")
global_MAX_LEN = 512        # max_len of scibert is 512

if __name__ == '__main__':
    train_data = utils.load_json(save_dir, "Train_papers_refs_pair.json")
    env = PSTEnv(text=True, graph=False, ranking=False)
    for ep in tqdm(range(100)):
        state, global_state = env.reset()
        while not env.done:
            action = torch.tensor([0.6])
            s, r, done, info, steps = env.step(action)
