import settings
import os
from os.path import join
import json
import utils
import pickle
from tqdm import tqdm
import time

save_dir = join(settings.PROJ_DIR, "data_processed")
train_data = utils.load_json(save_dir, "Train_papers_refs_pair.json")
valid_data = utils.load_json(save_dir, "Valid_papers_refs_pair.json")
train_keys = list(train_data.keys())
valid_keys = list(valid_data.keys())


def divide_llm_vec(global_states, local_state, name="scibert"):
    train_global_state_dic = {}
    train_local_state_dic = {}
    valid_global_state_dic = {}
    valid_local_state_dic = {}

    # divide llm vec data
    for key in tqdm(train_keys):
        train_global_state_dic[key] = global_states[key]
        train_local_state_dic[key] = local_state[key]

    for key in tqdm(valid_keys):
        valid_global_state_dic[key] = global_states[key]
        valid_local_state_dic[key] = local_state[key]

    # save train and valid data
    pickle.dump(train_global_state_dic, open(join(save_dir, "train_global_states_{}.pkl".format(name)), "wb"))
    pickle.dump(train_local_state_dic, open(join(save_dir, "train_local_states_{}.pkl".format(name)), "wb"))
    pickle.dump(valid_global_state_dic, open(join(save_dir, "valid_global_states_{}.pkl".format(name)), "wb"))
    pickle.dump(valid_local_state_dic, open(join(save_dir, "valid_local_states_{}.pkl".format(name)), "wb"))

    print("{} train and valid vec saved successfully!".format(name))


if __name__ == "__main__":
    # read llmvec data
    global_states = pickle.load(open(join(save_dir, "global_states_scibert.pkl"), "rb"))
    local_state = pickle.load(open(join(save_dir, "local_states_scibert.pkl"), "rb"))
    divide_llm_vec(global_states, local_state, name="scibert")

    global_states_llama = pickle.load(open(join(save_dir, "global_states_llama2.pkl"), "rb"))
    local_state_llama = pickle.load(open(join(save_dir, "local_states_llama2.pkl"), "rb"))
    divide_llm_vec(global_states_llama, local_state_llama, name="llama2")

    global_states_gemma = pickle.load(open(join(save_dir, "global_states_gemma2.pkl"), "rb"))
    local_state_gemma = pickle.load(open(join(save_dir, "local_states_gemma2.pkl"), "rb"))
    divide_llm_vec(global_states_gemma, local_state_gemma, name="gemma2")

