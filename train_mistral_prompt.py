#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
#    We modified the code based on Alpaca train.py. Author: Zheng Yuan, Hongyi Yuan

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import io
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import json
import warnings
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch.nn as nn
import copy
from copy import deepcopy


import os
# import wandb

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def cleanup():
    """Cleanup resources like GPU memory and distributed processes."""
    print("Cleaning up resources...")
    
    # Synchronize all distributed processes (if initialized)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()  # Synchronize all processes
        torch.distributed.destroy_process_group()  # Destroy the process group
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    print("Resources cleaned up.")
    
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

sys_template = """<<SYS>>\nYou are a helpful, respectful and honest assistant. 
                Always answer as helpfully as possible, while being safe.{}\n<</SYS>>\n\n {} """
    

# Define a custom Cross-Response Attention Mechanism
class CrossResponseAttention(nn.Module):
    def __init__(self, d_model,device,dtype, num_heads=4):
        super(CrossResponseAttention, self).__init__()
        self.query_proj = nn.Linear(d_model, d_model).to(device).to(dtype=dtype)
        self.response_proj = nn.Linear(d_model, d_model).to(device).to(dtype=dtype)
        self.attention_layer = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads).to(device).to(dtype=dtype)
        
    def forward(self, query_representation, response_representations):
        # query_representation: (1, seq_len, d_model)
        # response_representations: (num_responses, seq_len, d_model)
        
        # Project query and responses

        query_proj = self.query_proj(query_representation.unsqueeze(0)).permute(1, 0, 2)  # Shape: (seq_len, 1, d_model)
        response_proj = self.response_proj(response_representations).permute(1, 0, 2)  # Shape: (seq_len, num_responses, d_model)

        # Pass the tensors to the attention layer
        attention_output, attention_weights = self.attention_layer(
            query_proj, response_proj, response_proj
        )
        
        return attention_weights.squeeze(0), attention_output.squeeze(0)
            
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    sft_checkpoint: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    stop_response: bool = field(default=False)
    train_sample_num: int = field(default=2)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    length_penalty: float = field(default=1.0)
    only_use_provide: bool = field(default=False)
    only_use_sample: bool = field(default=False)
    train_method: str = field(default="dpo")
    # wandb_path: str = field(default=None)
    # report_to: str = field(default='wandb')


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class ScoreDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(ScoreDataset, self).__init__()
        # one can also experiment with different data loading here
        with open(data_path, "r") as f:
            lines = f.readlines()
        self.data = [json.loads(line.strip()) for line in lines][:40000]

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return dict(input_ids=self.data[i])


def _single_tokenize(text, tokenizer, max_len=None):
    if max_len is None:
        max_len = tokenizer.model_max_length
    toked = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        max_length=max_len,
        truncation=True,
    )
    return toked["input_ids"][0]


def stop_response(res):
    stops = ["\n\nHuman:", "\n\nAssistant:", "\n\nhuman:", "\n\nassistant:"]   ###this is for hh dataset, one can also experiment with different stop tokens here
    for stop in stops:
        if res.find(stop) >= 0:
            res = res[: res.find(stop)].strip()
    return res


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    stop_response: bool
    num: int
    data_path: str

    def __call__(self, instances):
        idxs = []
        all_scores = []
        input_ids = []
        all_probs = []
        labels = []
        query_len = []
        for idx, ins in enumerate(instances):
            ins = ins["input_ids"]  # hack
            query = ins["query"]

            responses = ins["responses"][-self.num:]
            scores = ins["scores"][-self.num:]
            probs = ins['probs'][-self.num:]

            all_scores.append(scores)
            all_probs.append(probs)
            idxs.append([idx] * len(scores))
            
            if "help" in self.data_path:
                prompt_input = '\n\nHuman: ' + query + '\n\nAssistant: '
            else:
                prompt_input = query

            #######if we want to add sys template manually#########################
            # prompt_input = sys_template.format_map(prompt_input)
            ###########################################################################
            
            ######keep end for prompt
            self.tokenizer.truncation_side = "left"

            query_input_ids = _single_tokenize(
                prompt_input,
                self.tokenizer,
                max_len=int(self.tokenizer.model_max_length * 2 / 3),
            )
            query_target = torch.LongTensor(
                [IGNORE_INDEX] * (query_input_ids.shape[0] - 1)
            )
            dummy_target = torch.LongTensor([IGNORE_INDEX])

            ##for responses, always keep start
            self.tokenizer.padding_side = "right"
            self.tokenizer.truncation_side = "right"

            for res in responses:
                if self.stop_response:
                    r = stop_response(res)
                else:
                    r = res
                res_input_ids = _single_tokenize(
                    r + self.tokenizer.eos_token,
                    self.tokenizer,
                    max_len=self.tokenizer.model_max_length - query_input_ids.shape[0],
                )  # eos here
                input_ids.append(torch.cat((query_input_ids, res_input_ids), dim=0))
                labels.append(
                    torch.cat((query_target, res_input_ids, dummy_target), dim=0)
                )
                query_len.append(len(query_input_ids))


        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=labels,
            idxs=torch.LongTensor(idxs),
            scores=torch.FloatTensor(all_scores),
            probs = torch.FloatTensor(all_probs),
            query_len=query_len,
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = ScoreDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, stop_response=data_args.stop_response,num=data_args.train_sample_num, data_path = data_args.data_path
    )
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )

import torch.distributed as dist
class DORA_Trainer(Trainer):

     # in case something goes wrong when saving the checkpoint, enforce the process to exit
    # def _save_checkpoint(self, model, trial,metrics):
    #     super()._save_checkpoint(model, trial,metrics)
    #     if dist.is_initialized():
    #         dist.barrier()  
    #     os._exit(0 if self.args.process_index == 0 else 1)

            
    def gather_logits_labels(self, logits, labels_):
        labels = labels_.clone()
        mask = (labels != -100).float()
        new_logits = logits.clone()  # Create a copy to avoid in-place modification
        labels[labels == -100] = 0
        output = torch.gather(new_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(
            -1
        )
        output = output * mask  # B * L
        return output

    def get_score(self, logit_label, labels):
        mask = (labels != -100).float()
        length = mask.sum(-1)
        scores = logit_label.sum(-1) / (length**self.args.length_penalty)
        return scores

    def rrhf_loss(self, scores, rw_scores,probs,logit_label,method):
        cand = rw_scores.shape[1]
        bz = rw_scores.shape[0]
        new_scores = scores.reshape(-1, cand)  # batch * cand
        diff = new_scores.unsqueeze(1) - new_scores.unsqueeze(-1)  # batch * cand * cand
        rw_diff = rw_scores.unsqueeze(1) - rw_scores.unsqueeze(-1)
        aval = torch.bitwise_and(rw_diff > 0, diff < 0)

        J = (diff*aval).sum(dim=-1).sum(dim=-1)/cand

        rrhf_loss=-J
        sft_loss = self.sft_loss(logit_label, rw_scores)
        losses = 1.0*(rrhf_loss.mean())+sft_loss
        lambda_ = 1.0
        if method=="rrhf":
            return losses.mean() 
        else:
            ######num=2
            # alpha=1/2
            # beta=1/2
            # alpha_ = torch.tensor([beta,alpha]).to(probs.device)
            
            # ###num=4:
            alpha = 1/4
            beta = 1/2 ###1/2 /2 
            alpha_ = torch.tensor([beta/2,beta/2,(1-alpha-beta),alpha]).to(probs.device)
            
            # ######num=6
            # alpha = 1/6
            # beta = 1/2 ###1/2 /2 
            # alpha_ = torch.tensor([beta/4,beta/4,beta/4,beta/4,(1-alpha-beta),alpha]).to(probs.device)
        
            denomi=torch.mul(alpha_,((1-probs)/probs))
            weight= 1/ (alpha+denomi)
            weight = torch.reshape(weight,(bz,-1))
          
            return lambda_*torch.log(torch.mean(torch.exp(losses*weight / lambda_)))

   
    def lire_loss(self, logit_label, rw_scores,probs, method):
        T = 2.0
        cand = rw_scores.shape[1]
        bz = rw_scores.shape[0]
        logit_label_batch = torch.reshape(
            logit_label, (-1, cand, logit_label.shape[-1])
        )  # batch * cand
        summed_logit = logit_label_batch.sum(-1)
        Q = (summed_logit / T).softmax(dim=-1)
        J = torch.mul(Q, rw_scores.softmax(dim=-1))
        losses = -J.mean(dim=-1)
        losses = torch.reshape(losses,(bz,-1))
        
        lambda_ = 1.0

        if method == "lire":
            return losses.mean()
        else:
            # ######num=2
            # alpha=1/2
            # beta=1/2
            # alpha_ = torch.tensor([beta,alpha]).to(probs.device)
            # # ######num=4   
            alpha = 1/4
            beta = 1/2 
            alpha_ = torch.tensor([beta/2,beta/2,(1-alpha-beta),alpha]).to(probs.device)
            
            
            denomi=torch.mul(alpha_,((1-probs)/probs))
            weight3= 1/ (alpha+denomi)
            return lambda_ *weight3.mean()*torch.log(torch.mean(torch.exp(losses/ lambda_)))


    def sft_loss(self, logit_label, rw_scores):
    
        cand = rw_scores.shape[1]
        logit_label_batch = torch.reshape(
            logit_label, (-1, cand, logit_label.shape[-1])
        )  # batch * cand * L
        expert_response_logit_label = logit_label_batch[
            torch.arange(rw_scores.shape[0]), -2
        ].squeeze()
        return -expert_response_logit_label.mean()

    def dpo_loss(self,logit_label, logit_label_base, rw_scores,probs,method):
        cand = rw_scores.shape[1]
        bz = rw_scores.shape[0]
        logit_label_batch = torch.reshape(
            logit_label, (-1, cand, logit_label.shape[-1])
        )  # batch * cand
        logit_label_base_batch = torch.reshape(
            logit_label_base, (-1, cand, logit_label.shape[-1])
        )  # batch * cand
        summed_logit = logit_label_batch.sum(-1)
        summed_logit_base = logit_label_base_batch.sum(-1)
        

        
        ####pairwise dpo under BT preference model###########################
        # policy_chosen_logps = summed_logit[:,0]
        # policy_rejected_logps = summed_logit[:,1]
        # reference_chosen_logps = summed_logit_base[:,0]
        # reference_rejected_logps = summed_logit_base[:,1]
        # pi_logratios = policy_chosen_logps - policy_rejected_logps
        # ref_logratios = reference_chosen_logps - reference_rejected_logps
        # logits = pi_logratios - ref_logratios
        # losses = -F.logsigmoid(0.1 * logits)
        ###list dpo under PL preference model################################
        # # # # ## Sort rw_scores and get the sorted indices
        sorted_rw_scores, sorted_indices = torch.sort(rw_scores, descending=True, dim=1)
        # Use sorted indices to reorder summed_logit
        sorted_summed_logit = torch.gather(summed_logit, 1, sorted_indices)
        sorted_based_summed_logit = torch.gather(summed_logit_base, 1, sorted_indices)
        logits = sorted_summed_logit -  sorted_based_summed_logit
        logits_exp = (0.1*logits).exp()
        flip_ = torch.flip(logits_exp,dims=(1,))
        suma = torch.cumsum(flip_,dim=-1)
        J = torch.log(flip_/suma)
        losses = -J.mean(dim=-1)

        losses = torch.reshape(losses, (bz,-1))

        lambda_=1.0
        
        if method == "dpo":
            return losses.mean()
        else:
            lambda_=1.0
            # #######num=2
            # alpha=1/2
            # beta=1/2
            # alpha_ = torch.tensor([beta,alpha]).to(probs.device)
        
            ######num=4
            alpha = 1/4
            beta = 1/2 ###1/2 /2 
            alpha_ = torch.tensor([beta/2,beta/2,alpha,(1-alpha-beta)]).to(probs.device)
            denomi=torch.mul(alpha_,((1-probs)/probs))
            weight= 1/ (alpha+denomi)
            weight = torch.mean(weight,dim=-1).reshape((bz,-1))

            return lambda_ *torch.log(torch.mean(torch.exp(losses*weight/ lambda_)))

    def load_ref_model(self,model,model_args,training_args):
    
        ref_model = deepcopy(model)
        self.base_model = ref_model
        self.base_model.eval()
        return None


    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        if self.args.only_use_provide:
            inputs["input_ids"] = inputs["input_ids"][-2:]
            inputs["attention_mask"] = inputs["attention_mask"][-2:]
            inputs["labels"] = inputs["labels"][-2:]
            inputs["idxs"] = inputs["idxs"][:, -2:]
            inputs["scores"] = inputs["scores"][:, -2:]
        if self.args.only_use_sample:
            inputs["input_ids"] = inputs["input_ids"][:-2]
            inputs["attention_mask"] = inputs["attention_mask"][:-2]
            inputs["labels"] = inputs["labels"][:-2]
            inputs["idxs"] = inputs["idxs"][:, :-2]
            inputs["scores"] = inputs["scores"][:, :-2]

        query_len = inputs['query_len'][-1]
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            output_hidden_states=True
        )
        logits = outputs[0] ###(batch * cand) * L * V

        logits = F.log_softmax(logits, dim=-1)
        logit_label = self.gather_logits_labels(logits, inputs.get("labels"))
        ######add ref model for dpo loss or other loss that require the reference model##################
        self.base_model = self.base_model.to(model.device)
        with torch.no_grad():
            logits_base = self.base_model(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
            )[0]
        logits_base_ = F.log_softmax(logits_base, dim=-1)
        logit_label_base = self.gather_logits_labels(logits_base_, inputs.get("labels"))
        ########################################################################
        method=self.args.train_method
        if 'rrhf' in method:
            scores = self.get_score(logit_label, inputs.get("labels"))
            loss = self.rrhf_loss(scores,  inputs.get("scores"),inputs.get('probs'),logit_label,method)
            return loss
        elif 'lire' in method:
             loss = self.lire_loss(logit_label, inputs.get("scores"),inputs.get('probs'),method)
        elif 'dpo' in method:
            loss = self.dpo_loss(logit_label,logit_label_base,inputs.get("scores"),inputs.get('probs'),method)
        return loss

def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.float16,
    )
    policy_checkpoint = model_args.sft_checkpoint
 
    ######loading the weights from the SFT checkpoint, can be modified to load the weights from the checkpoint of the reference model
    state_dict = torch.load(f"{policy_checkpoint}/policy.pt", map_location='cpu')
    model.load_state_dict(state_dict['state'])
    ###############################
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    ##### apply lora to llama
    # Define LoRA Config
    lora_config = LoraConfig(
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
        target_modules=["q_proj", "v_proj"],
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
    )

    # # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = DORA_Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module,
    )
      ####if DPO loss is applied, initialize the reference model
    trainer.load_ref_model(model,model_args,training_args)
    ##############################################################
    trainer.train()


if __name__ == "__main__":
    import numpy as np
    import random

    train()
