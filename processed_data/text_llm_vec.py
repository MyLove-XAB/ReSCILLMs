import numpy as np
from transformers import (BertForSequenceClassification, BertTokenizer, LlamaForSequenceClassification, LlamaTokenizer,
                          GemmaForSequenceClassification, GemmaTokenizer, AutoTokenizer)
import json
import pickle
import settings
import utils
from RLmethod.pstenv import extract_paper_text_info
import torch
from os.path import join
from tqdm import tqdm


save_dir = settings.PROJ_DIR + "/data_processed"
train_data = utils.load_json(save_dir, "Train_papers_refs_pair.json")
valid_data = utils.load_json(save_dir, "Valid_papers_refs_pair.json")
train_data_small = {key: train_data[key] for key in list(train_data.keys())[:4]}
valid_data_small = {key: valid_data[key] for key in list(valid_data.keys())[:4]}
scibert = settings.PROJ_DIR + "/pretrain_models/bertmodel"
llama = settings.PROJ_DIR + "/pretrain_models/llamamodel/llama2-7b-hf"
gemma = settings.PROJ_DIR + "/pretrain_models/gemma"


def get_global_llm_vec():
    """
    Get the global LLM vector for each paper in the training and validation set with scibert and save it to a file.
    Returns: global_state_dic

    """
    global_MAX_LEN = 512
    lm = BertForSequenceClassification.from_pretrained(scibert)
    lm.eval()
    for i, param in enumerate(lm.parameters()):
        if i < 201:
            param.requires_grad = False
    lm.cuda() if torch.cuda.is_available() else lm
    tokenizer = BertTokenizer.from_pretrained(scibert)
    emb_layer = lm.base_model.embeddings
    encoder = lm.base_model.encoder

    global_state_dic = dict()

    for key in tqdm(train_data.keys() | valid_data.keys()):
        query_paper = key
        global_info = extract_paper_text_info(query_paper, query_paper=query_paper, context_pos=False, content=True)

        query_text_input = "[SEP]".join(global_info)
        global_query_tokens = tokenizer.encode(query_text_input, padding="max_length", truncation=True,
                                                    return_tensors="pt", max_length=global_MAX_LEN)
        tmp = emb_layer(global_query_tokens.cuda() if torch.cuda.is_available() else global_query_tokens)
        global_state = encoder(tmp)

        global_state_dic[query_paper] = global_state.last_hidden_state.detach().cpu()[:, 0, :].numpy().tolist()

    # release memory
    del lm
    del tokenizer
    del emb_layer
    del encoder
    torch.cuda.empty_cache()

    pickle.dump(global_state_dic, open(join(save_dir, "global_states_scibert.pkl"), "wb"))
    return global_state_dic


def get_local_llm_vec():
    global_MAX_LEN = 512
    lm = BertForSequenceClassification.from_pretrained(scibert)
    lm.eval()
    for i, param in enumerate(lm.parameters()):
        if i < 201:
            param.requires_grad = False
    lm.cuda() if torch.cuda.is_available() else lm
    tokenizer = BertTokenizer.from_pretrained(scibert)
    emb_layer = lm.base_model.embeddings
    encoder = lm.base_model.encoder

    local_state_dic = dict()

    for key in tqdm(train_data.keys() | valid_data.keys()):
        local_state_dic[key] = dict()
        query_paper = key
        try:
            pairs = train_data[key]
        except:
            pairs = valid_data[key]

        for item in pairs:
            local_info = extract_paper_text_info(item[0], query_paper=query_paper, context_pos=item[1], content=False)
            local_text_input = "[SEP]".join([str(tokens) if tokens is not None else "" for tokens in local_info])
            local_query_tokens = tokenizer.encode(
                local_text_input, padding="max_length", truncation=True, return_tensors="pt", max_length=global_MAX_LEN)
            tmp = emb_layer(local_query_tokens.cuda() if torch.cuda.is_available() else local_query_tokens)
            # tmp = emb_layer(local_query_tokens)
            local_state = encoder(tmp)
            local_state_dic[key][item[1]] = local_state.last_hidden_state.detach().cpu()[:, 0, :].numpy()

    pickle.dump(local_state_dic, open(join(save_dir, "local_states_scibert.pkl"), "wb"))

    # release memory
    del lm
    del tokenizer
    del emb_layer
    del encoder
    torch.cuda.empty_cache()

    return local_state_dic


def get_local_llm_vec_llama():
    global_MAX_LEN = 5000
    num = 0
    lm = LlamaForSequenceClassification.from_pretrained(llama, torch_dtype=torch.float16)          # torch_dtype=torch.float16
    lm.eval()
    for i, param in enumerate(lm.parameters()):
        # print(i)
        if i < 291:
            param.requires_grad = False
    lm.cuda() if torch.cuda.is_available() else lm
    tokenizer = LlamaTokenizer.from_pretrained(llama)
    tokenizer.pad_token = tokenizer.eos_token

    local_state_dic = dict()
    for key in tqdm(train_data.keys() | valid_data.keys()):
        local_state_dic[key] = dict()
        query_paper = key
        try:
            pairs = train_data[key]
        except:
            pairs = valid_data[key]

        for item in pairs:
            local_info = extract_paper_text_info(item[0], query_paper=query_paper, context_pos=item[1], content=False)
            local_text_input = "[SEP]".join([str(tokens) if tokens is not None else "" for tokens in local_info])
            if local_text_input is None or local_text_input == "":
                local_text_input = ["[SEP]"]
                num += 1
            local_query_tokens = tokenizer.encode(
                local_text_input, truncation=True, return_tensors="pt", max_length=global_MAX_LEN)
            local_state = lm.base_model(local_query_tokens.cuda() if torch.cuda.is_available() else local_query_tokens)
            sequence_output = local_state.last_hidden_state
            cls_embedding = torch.mean(sequence_output, dim=1)
            # cls_embedding = sequence_output[:, -1, :]       # final hidden state
            tmp = cls_embedding.detach().cpu().numpy().tolist()
            local_state_dic[key][item[1]] = tmp

    print("Number of empty local text: ", num)
    pickle.dump(local_state_dic, open(join(save_dir, "local_states_llama2.pkl"), "wb"))

    # release memory
    del lm
    del tokenizer
    torch.cuda.empty_cache()

    return local_state_dic


def get_global_llm_vec_llama():
    global_MAX_LEN = 5000
    num = 0
    lm = LlamaForSequenceClassification.from_pretrained(llama, torch_dtype=torch.float16)
    lm.eval()
    for i, param in enumerate(lm.parameters()):
        print(i)
        if i < 291:
            param.requires_grad = False
    lm.cuda() if torch.cuda.is_available() else lm
    tokenizer = LlamaTokenizer.from_pretrained(llama)
    tokenizer.pad_token = tokenizer.eos_token

    global_state_dic = dict()
    for key in tqdm(train_data.keys() | valid_data.keys()):
        query_paper = key    # key
        global_info = extract_paper_text_info(query_paper, query_paper=query_paper, context_pos=False, content=True)

        query_text_input = "[SEP]".join(global_info)

        if query_text_input is None or query_text_input == "":
            query_text_input = ["[SEP]"]
            num += 1
        global_query_tokens = tokenizer.encode(query_text_input, truncation=True,
                                               return_tensors="pt", max_length=global_MAX_LEN)

        global_state = lm.base_model(global_query_tokens.cuda() if torch.cuda.is_available() else global_query_tokens)
        sequence_output = global_state.last_hidden_state
        cls_embedding = torch.mean(sequence_output, dim=1)          # mean pooling
        # cls_embedding = sequence_output[:, -1, :]       # final hidden state
        tmp = cls_embedding.detach().cpu().numpy().tolist()
        global_state_dic[query_paper] = tmp

    # release memory
    del lm
    del tokenizer
    torch.cuda.empty_cache()

    print("Number of empty global text: ", num)

    pickle.dump(global_state_dic, open(join(save_dir, "global_states_llama2.pkl"), "wb"))
    return global_state_dic


def get_global_llm_vec_gemma():
    num = 0
    lm = GemmaForSequenceClassification.from_pretrained(gemma, torch_dtype=torch.float16)
    lm.eval()
    for i, param in enumerate(lm.parameters()):
        print(i)
        if i < 165:
            param.requires_grad = False
    lm.cuda() if torch.cuda.is_available() else lm
    tokenizer = AutoTokenizer.from_pretrained(gemma)

    global_state_dic = dict()
    for key in tqdm(train_data.keys() | valid_data.keys()):
        query_paper = key    # key
        global_info = extract_paper_text_info(query_paper, query_paper=query_paper, context_pos=False, content=True)

        query_text_input = "[SEP]".join(global_info)

        if query_text_input is None or query_text_input == "":
            query_text_input = ["[SEP]"]
            num += 1
        global_query_tokens = tokenizer.encode(query_text_input,
                                               return_tensors="pt")

        global_state = lm.base_model(global_query_tokens.cuda() if torch.cuda.is_available() else global_query_tokens)
        sequence_output = global_state.last_hidden_state
        cls_embedding = torch.mean(sequence_output, dim=1)          # mean pooling
        # cls_embedding = sequence_output[:, -1, :]       # final hidden state
        tmp = cls_embedding.detach().cpu().numpy().tolist()
        # global_state_dic[query_paper] = global_state.last_hidden_state.detach().cpu()[:, 0, :].numpy().tolist()
        global_state_dic[query_paper] = tmp

    # release memory
    del lm
    del tokenizer
    torch.cuda.empty_cache()

    print("Number of empty global text: ", num)

    pickle.dump(global_state_dic, open(join(save_dir, "global_states_gemma2.pkl"), "wb"))


def get_local_llm_vec_gemma():
    num = 0
    lm = GemmaForSequenceClassification.from_pretrained(gemma, torch_dtype=torch.float16)
    lm.eval()
    for i, param in enumerate(lm.parameters()):
        print(i)
        if i < 165:
            param.requires_grad = False
    lm.cuda() if torch.cuda.is_available() else lm
    tokenizer = AutoTokenizer.from_pretrained(gemma)

    local_state_dic = dict()
    # for key in tqdm(train_data_small.keys() | valid_data_small.keys()):      # train_data.keys() | valid_data.keys()):
    for key in tqdm(train_data.keys() | valid_data.keys()):
        local_state_dic[key] = dict()
        query_paper = key
        try:
            pairs = train_data[key]
        except:
            pairs = valid_data[key]

        for item in pairs:
            local_info = extract_paper_text_info(item[0], query_paper=query_paper, context_pos=item[1], content=False)
            local_text_input = "[SEP]".join([str(tokens) if tokens is not None else "" for tokens in local_info])
            if local_text_input is None or local_text_input == "":
                local_text_input = ["[SEP]"]
                num += 1
            local_query_tokens = tokenizer.encode(
                local_text_input, return_tensors="pt")
            local_state = lm.base_model(local_query_tokens.cuda() if torch.cuda.is_available() else local_query_tokens)
            sequence_output = local_state.last_hidden_state
            cls_embedding = torch.mean(sequence_output, dim=1)
            tmp = cls_embedding.detach().cpu().numpy().tolist()
            local_state_dic[key][item[1]] = tmp

    print("Number of empty local text: ", num)
    pickle.dump(local_state_dic, open(join(save_dir, "local_states_gemma2.pkl"), "wb"))

    # release memory
    del lm
    del tokenizer
    torch.cuda.empty_cache()

    return local_state_dic


if __name__ == "__main__":
    get_global_llm_vec()
    get_local_llm_vec()
    get_global_llm_vec_llama()
    get_local_llm_vec_llama()
    get_global_llm_vec_gemma()
    get_local_llm_vec_gemma()


