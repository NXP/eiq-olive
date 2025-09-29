# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from olive.constants import Framework
from olive.data.registry import Registry
from itertools import chain

model_id = "Qwen/Qwen3-1.7B"
config = AutoConfig.from_pretrained(model_id)


def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return tokenizer(examples["text"])


# We hardcode the input names related to the KV cache for the KV dataloader
input_names = list(chain.from_iterable(
    [[f"past_key_values.{j}.key", f"past_key_values.{j}.value"] for j in range(config.num_hidden_layers)]
))



# Dataloader for INCQuantization, approach = static. In particular, we need to create and pass dummy past_key_values
# for the onnx decoder.
class KVDataloader:
    def __init__(self, dataset="NeelNanda/pile-10k", pad_max=196, batch_size=1, n_batches=1, sub_folder="train"):
        self.pad_max = pad_max
        self.batch_size = batch_size
        self.n_batches = n_batches
        dataset = load_dataset(dataset, split=sub_folder)
        dataset = dataset.map(tokenize_function, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )
        self.key_value_input_names = [key for key in input_names if (".key" in key) or (".value" in key)]
        self.use_cache = len(self.key_value_input_names) > 0

    def collate_batch(self, batch):
        input_ids_padded = []
        attention_mask_padded = []
        last_ind = []

        for text in batch:
            input_ids = text["input_ids"]
            pad_len = self.pad_max - input_ids.shape[0]
            last_ind.append(input_ids.shape[0] - 1)
            attention_mask = torch.ones(len(input_ids))
            input_ids = pad(input_ids, (0, pad_len), value=1)
            attention_mask = pad(attention_mask, (0, pad_len), value=0)
            input_ids_padded.append(input_ids)
            attention_mask_padded.append(attention_mask)
        return (torch.vstack(input_ids_padded), torch.vstack(attention_mask_padded)), torch.tensor(last_ind)

    def __iter__(self):
        try:
            for j, ((input_ids, attention_mask), last_ind) in enumerate(self.dataloader):
                if j >= self.n_batches:
                    return

                print(f"Passing batch #{j + 1}/{self.n_batches}")
                ort_input = {}
                ort_input["input_ids"] = input_ids[:, :-1].detach().cpu().numpy().astype("int64")
                ort_input["attention_mask"] = attention_mask[:, :-1].detach().cpu().numpy().astype("int64")

                # Add position_ids
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                ort_input["position_ids"] = position_ids[:, :-1].detach().cpu().numpy().astype("int64")
                if self.use_cache:
                    # Create dummy past_key_values for decoder
                    num_attention_heads = config.num_key_value_heads
                    embed_size_per_head = config.hidden_size // config.num_attention_heads
                    shape = (self.batch_size, num_attention_heads, 0, embed_size_per_head)
                    key_or_value = np.zeros(shape, dtype=np.float32)
                    for key_value_input_name in self.key_value_input_names:
                        ort_input[key_value_input_name] = key_or_value

                yield ort_input, last_ind

        except StopIteration:
            return


@Registry.register_dataloader()
def kv_dataloader(dataset, batch_size, n_batches, **kwargs):
    return KVDataloader(pad_max=196, batch_size=batch_size, n_batches=n_batches)


## Custom evaluation function for wikitext PPL
#
def eval_wt2_ppl(model, device, execution_provider, tasks=("wikitext",), batch_size=128):
    from neural_compressor.evaluation.lm_eval import evaluate, LMEvalParser # noqa: PLC0415

    model_dir = Path(model.model_path).resolve().parent
    tokenizer = "Qwen/Qwen3-1.7B"

    if model.framework == Framework.ONNX:
        output_config_file = model_dir / "config.json"
        config.to_json_file(output_config_file, use_diff=False)
        eval_args = LMEvalParser(
            model="hf",
            model_args=f"pretrained={model_dir},tokenizer=" + tokenizer + ",model_format=onnx",
            batch_size=batch_size,
            tasks=",".join(tasks),
            device="cpu",
            verbosity="DEBUG",
        )
        results = evaluate(eval_args)

    elif model.framework == Framework.PYTORCH:
        eval_args = LMEvalParser(
            model="hf",
            model_args=f"pretrained={model.model_path},tokenizer={tokenizer},dtype=float32",
            batch_size=batch_size,
            tasks=",".join(tasks),
            device="cpu",
            verbosity="DEBUG",
        )
        results = evaluate(eval_args)

    eval_acc = 0
    for task_name in tasks:
        if task_name == "wikitext":
            print("Accuracy for {} is: {}".format(task_name, results["results"][task_name]["word_perplexity,none"]))  # noqa: T201
            eval_acc += results["results"][task_name]["word_perplexity,none"]
        else:
            print("Accuracy for {} is: {}".format(task_name, results["results"][task_name]["acc,none"]))  # noqa: T201
            eval_acc += results["results"][task_name]["acc,none"]

    if len(tasks) != 0:
        eval_acc /= len(tasks)

    print(eval_acc)
    return {"custom": eval_acc}
