import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import settings
import utils
from torch.distributions import Categorical
from pstenv import PSTEnv
from os.path import join
# from processed_data.text_llm_vec import get_local_llm_vec, get_global_llm_vec, train_data_small, valid_data_small
import pickle
import json
from tqdm import tqdm
from Logtool import CreateLog
import time


class PPOmemory(object):
    def __init__(self, mini_batch_size):
        self.states = []    # 状态
        self.actions = []   # 实际采取的动作
        self.probs = []     # 动作概率
        self.vals = []      # critic输出的状态值
        self.rewards = []   # 奖励
        self.dones = []     # 结束标志

        self.mini_batch_size = mini_batch_size  # minibatch的大小

    def sample(self):
        n_states = len(self.states)  # memory记录数量=20
        batch_start = np.arange(0, n_states, self.mini_batch_size)  # 每个batch开始的位置[0,5,10,15]
        indices = np.arange(n_states, dtype=np.int64)  # 记录编号[0,1,2....19]
        np.random.shuffle(indices)  # 打乱编号顺序[3,1,9,11....18]
        mini_batches = [indices[i:i + self.mini_batch_size] for i in batch_start]
        # 生成4个minibatch，每个minibatch记录乱序且不重复，用于后续学习更新网络
        # state = np.array([np.array(x) for x in self.states])

        return (self.states,
                torch.tensor(np.array([tensor.numpy() for tensor in self.actions])),
                np.array(self.probs),
                np.array(self.vals),
                np.array(self.rewards),
                np.array(self.dones),
                mini_batches)

    # 每一步都存储trace到memory
    def push(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    # 固定步长更新完网络后清空memory
    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(feature_size*3, feature_size)
        self.fc2 = nn.Linear(feature_size, 2)
        # self.fc3 = nn.Linear(512, 128)
        # self.fc4 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化权重
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        # torch.nn.init.xavier_uniform_(self.fc3.weight)
        # torch.nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        # x = self.tanh(self.fc2(x))
        # x = self.tanh(self.fc3(x))
        x = self.fc2(x)
        return x


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(feature_size*3, feature_size)
        self.fc2 = nn.Linear(feature_size, 1)
        # self.fc3 = nn.Linear(512, 128)
        # self.fc4 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化权重
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        # torch.nn.init.xavier_uniform_(self.fc3.weight)
        # torch.nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        # x = self.tanh(self.fc2(x))
        # x = self.tanh(self.fc3(x))
        x = self.fc2(x)
        return x


# 定义Actor-Critic网络
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.text = True
        self.graph = False
        # 定义共同的隐藏层
        self.dropout = torch.nn.Dropout(0.5)

        self.tanh = torch.nn.Tanh()
        self.relu = nn.ReLU()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 定义Actor和Critic的分支
        self.actor = Actor()

        self.critic = Critic()

    def only_update(self, x):
        return self.actor(x), self.critic(x)

    def forward(self, x, signal="train"):
        x = self.get_state_embedding(x[0], x[1], signal=signal)
        x = torch.cat([x[0], x[1], x[2]], dim=-1)
        ac, v = self.only_update(x)
        return ac, v

    def get_action(self, state, signal="train"):
        logits_, value = self.forward(state, signal)
        probs = torch.softmax(logits_, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, probs.data.cpu().numpy()[0], value

    def get_global_embedding(self, paper, signal="train"):      # update global state
        # get the global state
        self.global_state = []
        # self.global_paper = paper
        # get the text embedding
        if signal == "train":
            self.global_state = train_global_states[paper]
            self.global_state = torch.tensor(self.global_state, dtype=torch.float32).to(self.device)
        else:
            self.global_state = valid_global_states[paper]
            self.global_state = torch.tensor(self.global_state, dtype=torch.float32).to(self.device)

        return self.global_state

    def get_state_embedding(self, previous_paper, cur_paper, signal="train"):
        if signal == "train":
            pre_state = torch.tensor(train_local_state[self.global_paper][previous_paper], dtype=torch.float32).to(self.device)
            cur_state = torch.tensor(train_local_state[self.global_paper][cur_paper], dtype=torch.float32).to(self.device)
        else:
            pre_state = torch.tensor(valid_local_state[self.global_paper][previous_paper], dtype=torch.float32).to(self.device)
            cur_state = torch.tensor(valid_local_state[self.global_paper][cur_paper], dtype=torch.float32).to(self.device)

        state = [self.global_state, pre_state, cur_state]

        return state


# PPO 算法类
class PPO(object):
    def __init__(self, env, mbsize, lr, gamma=0.99, clip_ratio=0.2, update_epochs=10):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = ActorCritic().to(self.device)
        self.optimizer1 = optim.Adam(self.actor_critic.actor.parameters(), lr=lr[0])
        self.optimizer2 = optim.Adam(self.actor_critic.critic.parameters(), lr=lr[1])
        self.memory = PPOmemory(mini_batch_size=mbsize)
        self.max_success_rate = 0

    def update(self):
        for _ in range(self.update_epochs):
            (states_arr, actions_arr, log_probs_old_arr, values_arr,
             rewards_arr, dones_arr, mini_batches) = self.memory.sample()

            # 计算GAE
            values = values_arr[:]  # ndarray, shape: (batch_size,)和vals_arr一样 没必要重复赋值
            advantage = np.zeros(len(rewards_arr), dtype=np.float32)
            for t in range(len(rewards_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards_arr) - 1):
                    a_t += discount * (rewards_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    # 第一个时间点不用乘lambda, 所以discount是1，从第二个时间点开始不断地累乘lambda
                    discount *= self.gamma * 0.95
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device)  # # ndarray, shape: (batch_size,)

            # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 标准化优势
            values = torch.tensor(values).to(self.device)
            for batch in mini_batches:
                states_emb_ls = []
                for b in batch:
                    s = states_arr[b]
                    tmp = self.actor_critic.get_state_embedding(s[0], s[1], signal="train")
                    tmp_ = torch.cat([tmp[0], tmp[1], tmp[2]], dim=-1)
                    states_emb_ls.append(tmp_)

                # s_in = []
                # for s_emb in states_emb_ls:
                #     # logit, value_new = self.actor_critic(s)
                #     tmp = self.actor_critic.only_update1(s_emb)
                #     s_in.append(tmp)
                # s_in = torch.cat(states_emb_ls, dim=0)

                log_probs_old = torch.tensor(log_probs_old_arr[batch]).cuda() if torch.cuda.is_available() else torch.tensor(log_probs_old_arr[batch])      # shape: [mini_batch_size, 1]
                # ac_list = actions_arr[batch]
                actions = actions_arr[batch].to(self.device)  # shape: [mini_batch_size, 1]

                state_ = torch.cat(states_emb_ls, dim=0)
                logits_, values_new = self.actor_critic.only_update(state_)

                probs = torch.softmax(logits_, dim=-1)
                dist = Categorical(probs)
                log_probs_new = dist.log_prob(actions)

                # 计算比率 r(θ)
                ratio = log_probs_new.exp() / log_probs_old.exp()

                # 计算 PPO 损失
                weighted_probs = advantage[batch] * ratio  # shape: [mini_batch_size, ]
                weighted_clip_probs = torch.clamp(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clip_probs).mean()
                # seems not have categorical cross-entropy loss

                returns = advantage[batch] + values[batch]

                critic_loss = (returns - values_new) ** 2
                critic_loss = critic_loss.mean()

                loss1 = actor_loss
                loss2 = 0.5 * critic_loss

                # 更新网络参数
                self.optimizer1.zero_grad()
                loss1.backward()
                self.optimizer1.step()

                self.optimizer2.zero_grad()
                loss2.backward()
                self.optimizer2.step()
        self.memory.clear()  # 经验遍历过n_epoch之后，用完之后立即清空，后面重新进行交互采样填充

    def train(self, n_iter=100, sample_size=128):
        res_dic = dict()
        num = 1
        test_rewards, train_rewards, test_map = [], [], []
        for episode in tqdm(range(n_iter)):
            # if episode % 10 == 0:
            #     # test
            #     self.actor_critic.eval()
            #     success_rate = self.env.test_env(self.actor_critic)
            #     print("iter: %s, test reward: %s" % (episode, success_rate))
            #     test_rewards.append(success_rate)
            #     torch.save(self.actor_critic.state_dict(), join(MODEL_DIR, "ppo_policy_iter{}.pth".format(episode)))

            self.actor_critic.train()
            find = 0
            steps = 0
            # t_train = time.time()
            log.info("-------------START TRAINING----------------")
            for samp in range(sample_size):
                state, global_state = self.env.reset()
                # global_state = list(global_states.keys())[0]
                self.actor_critic.global_paper = global_state
                self.actor_critic.get_global_embedding(global_state, signal="train")

                done = False

                while not done:
                    # 与环境交互
                    steps += 1
                    action, log_prob, p, value = self.actor_critic.get_action(state, signal="train")          # , p
                    next_state, reward, done, info_, step_ = self.env.step(action)
                    # print(id(state))
                    # print(id(next_state))
                    # next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
                    # _, value = self.actor_critic(state)
                    log_ = torch.squeeze(log_prob).item()

                    # 存储数据
                    self.memory.push(state, torch.tensor(action),           # .to(self.device)
                                     log_, value.item(), reward, torch.tensor(done))
                    state = next_state
                self.update()

                # 计算价值函数的目标
                # _, next_value = self.actor_critic(state)
                # returns = self.compute_returns(rewards, dones, values, next_value.detach())
                if self.env.finally_found == 1:
                    find += 1

            # 打印每回合的总奖励
            # print(f"Episode {episode+1}, Find Rate: {find/sample_size}")
            # t_train_end = time.time()
            # log.info("--------------END TRAINING----------------, time spent: {} s, {} min".format(t_train_end - t_train, (t_train_end - t_train) / 60))

            train_rewards.append(find/sample_size)

            # peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            # log.info(f"Peak GPU memory during training: {peak_memory_gb:.2f} GB")

            if episode % TEST_EPOCH == 0:
                # test
                self.actor_critic.eval()
                # t_test = time.time()
                log.info("--------------START TEST----------------")
                success_rate, map = self.env.test_env(self.actor_critic)
                # t_test_end = time.time()
                # log.info("--------------END TEST----------------, time spent: {} s, {} min".format(t_test_end-t_test, (t_test_end-t_test)/60))
                print("iter: %s, test reward: %s, test map: %s" % (episode, success_rate, map))
                log.info("iter: %s, test reward: %s, test map: %s" % (episode, success_rate, map))
                test_rewards.append(success_rate)
                test_map.append(map)
                if self.max_success_rate < success_rate:
                    self.max_success_rate = success_rate
                    torch.save(self.actor_critic.state_dict(), join(MODEL_DIR, MODEL+"_policy_shuffle_run_{}.pth".format(num)))

        res_dic["iter"] = list(range(n_iter))
        res_dic["train_reward"] = train_rewards
        res_dic["test_reward"] = test_rewards
        res_dic["test_map"] = test_map
        with open(join(result_dir, MODEL+"_shuffle_result_{}.json".format(num)), "w") as f:
            json.dump(res_dic, f)


def case():
    # ppo = PPO(env, mbsize=MBSIZE, lr=LR, update_epochs=2)
    for i in range(1, 11):
        try:
            ppo.actor_critic.load_state_dict(torch.load(join(MODEL_DIR, MODEL + "_policy_shuffle_run_{}.pth".format(i))))
            print("load model: ", i)
            ppo.actor_critic.eval()
            success_rate, map = env.case(ppo.actor_critic)

            with open(join(save_dir, "case_result_{}.json".format(env.test_iter)), "w") as f:
                json.dump(env.case_result, f)
            print("test reward: %s, test map: %s" % (success_rate, map))
            log.info("test reward: %s, test map: %s" % (success_rate, map))
        except:
            pass


if __name__ == "__main__":
    # scibert = join(settings.PROJ_DIR, "pretrain_models/bertmodel")
    # llama2 = join(settings.PROJ_DIR, "pretrain_models/llmamodel/llama2-7b-hf")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()  # ⭐ 关键：清零历史峰值

    save_dir = join(settings.PROJ_DIR, "data_processed")
    MODEL_DIR = join(settings.PROJ_DIR, "saved_models")
    result_dir = join(settings.PROJ_DIR, "out")
    MODEL = "scibert"        # scibert, llama2, gemma2
    print("MODEL: ", MODEL)
    log = CreateLog(name="rose_log", filename=join(result_dir, MODEL+"_shuffle_log_1.log"), t_stamp=False, add_fh=True)

    if MODEL == "scibert":
        train_global_states = pickle.load(open(join(save_dir, "train_global_states_scibert.pkl"), "rb"))
        valid_global_states = pickle.load(open(join(save_dir, "valid_global_states_scibert.pkl"), "rb"))
        train_local_state = pickle.load(open(join(save_dir, "train_local_states_scibert.pkl"), "rb"))
        valid_local_state = pickle.load(open(join(save_dir, "valid_local_states_scibert.pkl"), "rb"))
        feature_size = 768
        LR = [8e-7, 8e-7]
        N_ITER, SAMPLE_SIZE = 1000, 788
        TEST_EPOCH = 1
        MBSIZE = 32
        HALF = False
    elif MODEL == "llama2":
        train_global_states = pickle.load(open(join(save_dir, "train_global_states_llama2.pkl"), "rb"))
        valid_global_states = pickle.load(open(join(save_dir, "valid_global_states_llama2.pkl"), "rb"))
        train_local_state = pickle.load(open(join(save_dir, "train_local_states_llama2.pkl"), "rb"))
        valid_local_state = pickle.load(open(join(save_dir, "valid_local_states_llama2.pkl"), "rb"))
        feature_size = 4096
        LR = [7e-7, 7e-7]
        N_ITER, SAMPLE_SIZE = 600, 788
        TEST_EPOCH = 1
        MBSIZE = 32
    else:
        train_global_states = pickle.load(open(join(save_dir, "train_global_states_gemma2.pkl"), "rb"))
        valid_global_states = pickle.load(open(join(save_dir, "valid_global_states_gemma2.pkl"), "rb"))
        train_local_state = pickle.load(open(join(save_dir, "train_local_states_gemma2.pkl"), "rb"))
        valid_local_state = pickle.load(open(join(save_dir, "valid_local_states_gemma2.pkl"), "rb"))
        feature_size = 2048
        LR = [7e-7, 7e-7]
        N_ITER, SAMPLE_SIZE = 600, 788
        TEST_EPOCH = 1
        MBSIZE = 32
    SHUFFLE_REFS = True
    SHUFFLE_SEED = 915
    SHUFFLE_VALID = True

    # global_states = get_global_llm_vec()
    # local_state = get_local_llm_vec()           # get local state

    env = PSTEnv(
        text=True,
        graph=False,
        shuffle_refs=SHUFFLE_REFS,
        shuffle_seed=SHUFFLE_SEED,
        shuffle_valid=SHUFFLE_VALID
    )

    ppo = PPO(env, mbsize=MBSIZE, lr=LR, update_epochs=2)

    ppo.train(n_iter=N_ITER, sample_size=SAMPLE_SIZE)

    # case()
