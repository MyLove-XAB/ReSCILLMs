import torch
import torch.nn as nn
import numpy as np
import settings
import utils
from torch.distributions import Categorical
from pstenv import PSTEnv
from os.path import join
import json
from tqdm import tqdm
from Logtool import CreateLog
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
import time


class PPOmemory(object):
    def __init__(self, mini_batch_size):
        self.states = []  # 状态
        self.actions = []  # 实际采取的动作
        self.probs = []  # 动作概率
        self.vals = []  # critic输出的状态值
        self.rewards = []  # 奖励
        self.dones = []  # 结束标志

        self.mini_batch_size = mini_batch_size  # minibatch的大小

    def sample(self):
        n_states = len(self.states)  # memory记录数量
        batch_start = np.arange(0, n_states, self.mini_batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        mini_batches = [indices[i:i + self.mini_batch_size] for i in batch_start]

        return (self.states,
                torch.tensor(np.array([tensor.numpy() for tensor in self.actions])),
                np.array(self.probs),
                np.array(self.vals),
                np.array(self.rewards),
                np.array(self.dones),
                mini_batches)

    def push(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

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
        self.fc1 = nn.Linear(feature_size * 3, feature_size, dtype=torch.bfloat16)
        self.fc2 = nn.Linear(feature_size, 2, dtype=torch.bfloat16)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(feature_size * 3, feature_size, dtype=torch.bfloat16)
        self.fc2 = nn.Linear(feature_size, 1, dtype=torch.bfloat16)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.text = True
        self.graph = False
        self.dropout = torch.nn.Dropout(0.5)
        self.max_length = MAX_LENGTH

        self.tanh = torch.nn.Tanh()
        self.relu = nn.ReLU()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor()
        self.critic = Critic()

        self.llm = model
        self.tokenizer = tokenizer

        # 用于记录当前episode的信息
        self.global_paper = None
        self.global_text = None

    def only_update(self, x):
        return self.actor(x), self.critic(x)

    def forward(self, x, signal="train"):
        # 仅用于非batch的前向传播（如采样时）
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
        return action.item(), log_prob, probs.data.cpu().to(torch.float16).numpy()[0], value

    def encode_texts(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        ).to(self.device)

        outputs = self.llm(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # [B, L, D]

        mask = inputs['attention_mask']
        last_idx = mask.sum(dim=1) - 1
        batch_idx = torch.arange(last_idx.size(0), device=self.device)
        return hidden_states[batch_idx, last_idx]  # [B, D]

    def get_global_embedding(self, paper, signal="train"):
        # 采样时只更新文本，不再计算embedding，embedding延迟到forward或batch update时计算
        self.global_text = global_paper_text[paper]
        pass

    def get_emb(self, papers):
        # 单条处理逻辑（用于采样过程）
        self.pre_text = local_paper_text[self.global_paper][papers[0]]
        self.cur_text = local_paper_text[self.global_paper][papers[1]]
        # 这里的 global_text 在 update 循环里可能会因为梯度问题需要重新获取
        # 采样阶段 forward 直接用当前的 global_text
        tmp = self.encode_texts([self.pre_text, self.cur_text, self.global_text])

        self.pre_emb = tmp[0:1]
        self.cur_emb = tmp[1:2]
        self.global_emb = tmp[2:3]

        return self.pre_emb, self.cur_emb, self.global_emb

    def get_state_embedding(self, previous_paper, cur_paper, signal="train"):
        self.get_emb([previous_paper, cur_paper])
        state = [self.global_emb, self.pre_emb, self.cur_emb]
        return state

    # 【新增功能】批量去重计算 Embedding
    def get_batch_state_embeddings(self, pre_papers, cur_papers, global_paper_text_val):
        """
        Input:
            pre_papers: list of IDs
            cur_papers: list of IDs
            global_paper_text_val: string
        Output:
            Tensor [Batch, Hidden*3]
        """
        batch_size = len(pre_papers)

        unique_text_map = {}  # text -> index
        unique_texts = []  # list of texts to encode

        # 索引矩阵 [Batch, 3] -> 对应 (pre, cur, global) 在 unique_texts 中的下标
        map_indices = torch.zeros((batch_size, 3), dtype=torch.long).to(self.device)

        for i in range(batch_size):
            t_pre = local_paper_text[self.global_paper][pre_papers[i]]
            t_cur = local_paper_text[self.global_paper][cur_papers[i]]
            t_glob = global_paper_text_val

            triplet = [t_pre, t_cur, t_glob]

            for col, text in enumerate(triplet):
                if text not in unique_text_map:
                    unique_text_map[text] = len(unique_texts)
                    unique_texts.append(text)
                map_indices[i, col] = unique_text_map[text]

        unique_embeddings = self.encode_texts(unique_texts)  # [Unique_N, Hidden]

        flatten_indices = map_indices.view(-1)
        # index_select 也是可导的，梯度会自动累加
        flatten_embeddings = unique_embeddings.index_select(0, flatten_indices)

        batch_embeddings = flatten_embeddings.view(batch_size, 3, -1)

        state_tensor = torch.cat([
            batch_embeddings[:, 2, :],  # Global
            batch_embeddings[:, 0, :],  # Pre
            batch_embeddings[:, 1, :]  # Cur
        ], dim=-1)

        return state_tensor


class PPO(object):
    def __init__(self, env, mbsize, lr, gamma=0.99, clip_ratio=0.2, update_epochs=10):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = ActorCritic().to(self.device)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.actor_critic.parameters()),
            lr=lr[0]
        )

        self.memory = PPOmemory(mini_batch_size=mbsize)
        self.max_success_rate = 0

    def update(self):
        for _ in range(self.update_epochs):
            (states_arr, actions_arr, log_probs_old_arr, values_arr,
             rewards_arr, dones_arr, mini_batches) = self.memory.sample()

            # 计算GAE
            values = values_arr[:]
            advantage = np.zeros(len(rewards_arr), dtype=np.float32)
            for t in range(len(rewards_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards_arr) - 1):
                    a_t += discount * (
                                rewards_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * 0.95
                advantage[t] = a_t

            advantage = torch.tensor(advantage).to(self.device)

            values = torch.tensor(values).to(self.device)

            for batch in mini_batches:
                batch_pre_papers = [states_arr[b][0] for b in batch]
                batch_cur_papers = [states_arr[b][1] for b in batch]

                current_global_text = self.actor_critic.global_text

                state_emb_batch = self.actor_critic.get_batch_state_embeddings(
                    batch_pre_papers,
                    batch_cur_papers,
                    current_global_text
                )

                log_probs_old = torch.tensor(log_probs_old_arr[batch]).to(self.device)
                actions = torch.tensor(actions_arr[batch]).to(self.device)

                adv_batch = advantage[batch]
                # adv_batch = (adv_batch - adv_batch.mean())/(adv_batch.std() + 1e-8)

                logits_, values_new = self.actor_critic.only_update(state_emb_batch)
                values_new = values_new.squeeze(-1)  # [B, 1] -> [B]

                probs = torch.softmax(logits_, dim=-1)
                dist = Categorical(probs)
                log_probs_new = dist.log_prob(actions)

                # 计算 Loss
                ratio = (log_probs_new - log_probs_old).exp()
                weighted_probs = adv_batch * ratio
                weighted_clip_probs = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_batch
                actor_loss = -torch.min(weighted_probs, weighted_clip_probs).mean()

                returns = adv_batch + values[batch]
                critic_loss = (returns - values_new) ** 2
                critic_loss = critic_loss.mean()

                loss = actor_loss + 0.5 * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()

                # 可选：手动清理缓存，防止显存碎片
                # torch.cuda.empty_cache()

        self.memory.clear()

    def train(self, n_iter=100, sample_size=128):
        res_dic = dict()
        test_rewards, train_rewards, test_map = [], [], []
        for episode in tqdm(range(n_iter)):
            self.actor_critic.train()
            find = 0
            steps = 0
            t_train = time.time()
            log.info("-------------START TRAINING----------------")
            # 采样阶段保持不变
            for samp in range(sample_size):
                with torch.no_grad():
                    state, global_state = self.env.reset()
                    self.actor_critic.global_paper = global_state
                    self.actor_critic.get_global_embedding(global_state)

                    done = False
                    while not done:
                        steps += 1
                        # 采样依然是一步步走，因为环境交互是序列的
                        action, log_prob, p, value = self.actor_critic.get_action(state, signal="train")
                        next_state, reward, done, info_, step_ = self.env.step(action)

                        log_ = torch.squeeze(log_prob).item()
                        self.memory.push(state, torch.tensor(action),
                                         log_, value.item(), reward, torch.tensor(done))
                        state = next_state

                # log.info("---------------START UPDATE----------------")
                # t_update = time.time()
                self.update()
                # t_update_end = time.time()
                # log.info("----------------END UPDATE----------------, time spent: {} s, {} min".format(t_update_end-t_update, (t_update_end-t_update)/60))
                if self.env.finally_found == 1:
                    find += 1
            t_train_end = time.time()
            log.info("--------------END TRAINING----------------, time spent: {} s, {} min".format(t_train_end-t_train, (t_train_end-t_train)/60))
            train_rewards.append(find / sample_size)

            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            log.info(f"Peak GPU memory during training: {peak_memory_gb:.2f} GB")

            if episode % TEST_EPOCH == 0:
                self.actor_critic.eval()
                t_test = time.time()
                log.info("--------------START TEST----------------")
                with torch.no_grad():
                    success_rate, map_score = self.env.test_env(self.actor_critic)
                t_test_end = time.time()
                log.info("--------------END TEST----------------, time spent: {} s, {} min".format(t_test_end-t_test, (t_test_end-t_test)/60))
                print("iter: %s, test reward: %s, test map: %s" % (episode, success_rate, map_score))
                log.info("iter: %s, test reward: %s, test map: %s" % (episode, success_rate, map_score))
                test_rewards.append(success_rate)
                test_map.append(map_score)

                if self.max_success_rate < success_rate:
                    self.max_success_rate = success_rate
                    # 保存 LoRA 和 Head
                    self.actor_critic.llm.save_pretrained(join(MODEL_DIR, MODEL + "_lora_policy_run_1"))
                    torch.save({
                        'actor_head': self.actor_critic.actor.state_dict(),
                        'critic_head': self.actor_critic.critic.state_dict(),
                    }, join(MODEL_DIR, MODEL + "_Heads_run_1.pth"))

        res_dic["iter"] = list(range(n_iter))
        res_dic["train_reward"] = train_rewards
        res_dic["test_reward"] = test_rewards
        res_dic["test_map"] = test_map
        with open(join(result_dir, MODEL + "_lora_result_1.json"), "w") as f:
            json.dump(res_dic, f)

def case():
    # ppo = PPO(env, mbsize=MBSIZE, lr=LR, update_epochs=2)

    for i in range(1, 11):
        try:
            ppo.actor_critic.llm = PeftModel.from_pretrained(
                base_model,
                join(MODEL_DIR, MODEL+"_lora_policy_run_{}".format(i)),  # 你保存的路径
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            checkpoint = torch.load(join(MODEL_DIR, MODEL + "_Heads_run_{}.pth".format(i)), map_location=DEVICE)
            ppo.actor_critic.actor.load_state_dict(checkpoint['actor_head'])
            ppo.actor_critic.critic.load_state_dict(checkpoint['critic_head'])

            print("load model: ", i)
            ppo.actor_critic.eval()
            with torch.no_grad():
                success_rate, map = env.case(ppo.actor_critic)

            with open(join(save_dir, "case_result_{}.json".format(env.test_iter)), "w") as f:
                json.dump(env.case_result, f)
            print("test reward: %s, test map: %s" % (success_rate, map))
            log.info("test reward: %s, test map: %s" % (success_rate, map))
        except:
            pass


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()  # ⭐ 关键：清零历史峰值
    scibert = join(settings.PROJ_DIR, "pretrain_models/bertmodel")
    llama2 = join(settings.PROJ_DIR, "pretrain_models/llmamodel/llama2-7b-hf")
    gemma2 = join(settings.PROJ_DIR, "pretrain_models/gemma")
    save_dir = join(settings.PROJ_DIR, "data_processed")
    MODEL_DIR = join(settings.PROJ_DIR, "saved_models")
    result_dir = join(settings.PROJ_DIR, "out")

    # 全局变量，ActorCritic 需要访问
    global_paper_text = utils.load_json(save_dir, "global_paper_text.json")
    local_paper_text = utils.load_json(save_dir, "local_paper_text.json")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL = "gemma"
    model_name = gemma2
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    # LoRA Config
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(base_model, lora_config)
    model = model.to(device=DEVICE)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    model.print_trainable_parameters()
    MAX_LENGTH = 512

    print("MODEL: ", model_name)
    log = CreateLog(name="rose_log", filename=join(result_dir, MODEL + "_lora_log_1.log"), t_stamp=False, add_fh=True)

    if MODEL == "gemma":
        feature_size = 2048
        LR = [1e-5, 1e-5]
        N_ITER, SAMPLE_SIZE = 100, 10
        TEST_EPOCH = 1
        MBSIZE = 16  # 可以尝试调大一点了，比如 8 或 16，视显存情况而定
    else:
        feature_size = 4096
        LR = [7e-7, 7e-7]
        N_ITER, SAMPLE_SIZE = 600, 788
        TEST_EPOCH = 1
        MBSIZE = 32

    env = PSTEnv(text=True, graph=False)

    ppo = PPO(env, mbsize=MBSIZE, lr=LR, update_epochs=2)
    ppo.train(n_iter=N_ITER, sample_size=SAMPLE_SIZE)