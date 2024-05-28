# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
from time import perf_counter
from typing import Dict, List, Optional, Union

import numpy as np
import transformers
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.logging_utils import logger

io_files = []
# vocab_size = 50257
# vocab_size = 32003

def write_io_files(
        inputs: Dict[str, np.ndarray],
        outputs: Dict[str, np.ndarray],
        write_io_dir: str,
        write_io_subdir: str,
        write_io_name: str,
        include_dims: bool = False,
        reset: bool = False,
):
    global io_files
    if reset:
        io_files = []
    io = []
    os.makedirs(f"{write_io_dir}/{write_io_subdir}", exist_ok=True)
    for iname, iarray in inputs.items():
        iarray.tofile(f"{write_io_dir}/{write_io_subdir}/{iname}.raw")
        ispec = {
            "path": f"{write_io_subdir}/{iname}.raw",
            "io-direction": "in",
            "elem-size": iarray.itemsize,
            "map-to": iname,
        }
        if include_dims:
            ispec["dims"] = iarray.shape
        io.append(ispec)
    for oname, oarray in outputs.items():
        oarray.tofile(f"{write_io_dir}/{write_io_subdir}/{oname}.raw")
        ospec = {
            "path": f"{write_io_subdir}/{oname}.raw",
            "io-direction": "out",
            "elem-size": oarray.itemsize,
            "map-to": oname,
        }
        if include_dims or oname.endswith("_RetainedState"):
            ospec["dims"] = oarray.shape
        io.append(ospec)
    io_files.append(io)
    with open(f"{write_io_dir}/{write_io_name}.json", "w") as fp:
        json.dump({"IO-files": io_files}, fp, indent=True)

def set_logits_bsize(session, batch_size, vocab_size):
    """
    sets the size of the output logits expected
    """
    # FIXME remove the vocab_size hardcoded value
    logits_out_placeholder = np.zeros((batch_size,vocab_size), dtype=np.float32)
    session.set_buffers({"logits": logits_out_placeholder})


def populate_inputs(source, dest, index):
    """
    populates the dest input dict at the specified index with the source input dict's items
    """
    for k, v in dest.items():
        # print("populating input at key ", k)
        if k == "batch_index":
            continue
        dest[k][index] = source[k]


def run_prefill(index, prefill_queue, session, tokenizer, prompt_len, ctx_len, decode_batch_size, slot_idx, write_io_dir=None):
    """
    runs prefill on the prompt at the specified index in the prefill queue

    returns the generated token id to start decoding from,
    the length of the prompt, the position id and cache index to
    start decoding this prompt from

    accepts decode batch size to populate the attention mask accordingly
    accepts the slot_idx to indicate which slot we are trying to replace
    with the current prompt/request
    """
    assert slot_idx < decode_batch_size
    decode_start_input = dict()
    # retrieve the prompt from the prefill queue
    prompt = prefill_queue[index]
    input_len = tokenizer(prompt, return_tensors="np", padding=True).input_ids.shape[1]
    num_chunks = -(input_len // -prompt_len)  # ceil divide without float
    input_len = num_chunks * prompt_len  # Convert input_len to a multiple of prompt_len
    assert input_len <= ctx_len, "input_len should be less than ctx_len"
    # pad the prompt tokens to match the input_len
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=input_len)
    # TODO need to store the attention mask and position ids for each batch element so that we can access them
    # at decode time
    inputs["attention_mask"] = np.concatenate(
        [inputs["attention_mask"].astype(bool) for j in range(decode_batch_size)], 0)
        
    inputs["position_ids"] = (np.cumsum(inputs["attention_mask"][0:1], 1) - 1) * inputs["attention_mask"][0:1]
    inputs["attention_mask"] = np.concatenate(
        [inputs["attention_mask"].astype(bool),np.zeros((decode_batch_size, ctx_len - input_len), dtype=bool),], 1,)
    
    cache_index = np.array([[0]], np.int64)
    batch_index = np.array([[slot_idx]], np.int64)
    inputs["cache_index"] = cache_index
    inputs["batch_index"] = batch_index

    # Run prefill
    for i in range(num_chunks):
        chunk_inputs = inputs.copy()
        chunk_inputs["input_ids"] = inputs["input_ids"][:, cache_index[0, 0]: cache_index[0, 0] + prompt_len]
        chunk_inputs["position_ids"] = inputs["position_ids"][:, cache_index[0, 0]: cache_index[0, 0] + prompt_len]
        chunk_inputs["attention_mask"] = inputs["attention_mask"].copy()
        chunk_inputs["attention_mask"][:, cache_index[0, 0] + prompt_len:] = False
        outputs = session.run(chunk_inputs)
        cache_index += prompt_len
        if write_io_dir:
            write_io_files(inputs, outputs, write_io_dir, "prefill", "aic_batch_io", True, False)

    # Get first token
    logits = outputs["logits"]
    if len(logits.shape) == 2:
        logits = np.expand_dims(logits, 1)
    decode_start_input_id = logits.argmax(2)
    decode_start_pos_id = inputs["attention_mask"][0:1].sum(1, keepdims=True)


    # Update the attention mask so that it can be used as input to start decode
    inputs["attention_mask"][:, cache_index] = True
    # populate the decode start info to return, to be able resume with the AR decoding this sequence
    # decoding will start using these inputs for the current request
    decode_start_input["input_ids"] = decode_start_input_id
    decode_start_input["position_ids"] = decode_start_pos_id
    decode_start_input["attention_mask"] = inputs.pop("attention_mask")[0:1]
    decode_start_input["cache_index"] = cache_index
    decode_start_input["batch_index"] = batch_index
    decode_start_input["input_len"] = input_len
    # print(f"prefill output id:{decode_start_input_id[0]} token: {tokenizer.convert_ids_to_tokens(decode_start_input_id[0])}")
    # print("returning decode start info as:", decode_start_input)

    return decode_start_input

def create_decode_inputs(decode_batch_size, tokenizer, cache_index, batch_index, ctx_len):
    """
    This function creates the decode inputs.

    Returns:
        dict: The decode inputs.
    """
    decode_inputs = {}
    # Create position IDs filled with zeros
    decode_inputs["position_ids"] = np.zeros((decode_batch_size, 1), np.int64)
    # Create input IDs filled with the pad token ID
    decode_inputs["input_ids"] = np.full((decode_batch_size, 1), tokenizer.pad_token_id)
    decode_inputs["cache_index"] = cache_index
    decode_inputs["batch_index"] = batch_index
    # Create attention mask filled with zeros
    decode_inputs["attention_mask"] = np.zeros((decode_batch_size, ctx_len), dtype=bool)
    return decode_inputs

def update_decode_inputs(decode_inputs, idx, input_lengths, req_id, generated_ids, next_token_id):
    """
    This function updates the decode inputs.

    Returns:
        dict: The updated decode inputs.
    """
    # Update the attention mask
    decode_inputs["attention_mask"][idx][
    input_lengths[req_id]: input_lengths[req_id] + len(generated_ids[req_id])] = 1
    decode_inputs["input_ids"][idx] = next_token_id[idx]
    # Increment the position IDs
    decode_inputs["position_ids"][idx] = decode_inputs["position_ids"][idx] + 1
    # Increment the cache index
    decode_inputs["cache_index"][idx] = decode_inputs["cache_index"][idx] + 1
    return decode_inputs

def get_decode_input(idx, prefill_queue, session, tokenizer, prompt_len, ctx_len, decode_batch_size, decode_inputs,
                     current_batch_req_ids, input_lengths, num_prompts_processed, generated_ids, write_io_dir):
    """
    This function runs prefill queue and populates the decode inputs.

    Returns:
        tuple: Updated prefill queue, decode inputs, current batch request IDs, input lengths, number of prompts processed, and generated IDs.
    """

    # Run prefill and get the start input for decode
    # FIXME assumes that prefill queue will always be popped from the front
    decode_start_input = run_prefill(
        index=0,
        prefill_queue=prefill_queue,
        session=session,
        tokenizer=tokenizer,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        decode_batch_size=decode_batch_size,
        slot_idx=idx,
        write_io_dir=write_io_dir)
    # TODO move below code to a function, that takes the prefill-dict and populates it into a particular index
    # this way, we will directly get the updated full batch input dict to run decode
    # print("Populating inputs to kick off decode for slot",idx)
    populate_inputs(source=decode_start_input, dest=decode_inputs, index=idx)
    # FIXME assumes that prefill queue will always be popped from the front
    current_batch_req_ids.append(idx)
    input_lengths[current_batch_req_ids[idx]] = decode_start_input["input_len"]
    num_prompts_processed += 1
    # update generated id list for this request, right after running prefill
    generated_ids[current_batch_req_ids[idx]].append(decode_start_input["input_ids"][0, 0])
    # pop the front of the prefill queue
    # assumes that prefill queue will always be popped from the front
    prefill_queue = prefill_queue[1:]
    return prefill_queue, decode_inputs, current_batch_req_ids, input_lengths, num_prompts_processed, generated_ids

def calculate_latency(generated_ids, decode_batch_size, loop_start, start, end, verbose):
    """
    This method will calucate the latency metrics
    
    Returns:
    total_num_decoded_tokens, prefill_perf, decode_perf, total_perf
    """

    total_num_decoded_tokens = sum([(len(generated_ids[i]) - 1) for i in range(decode_batch_size)])
    prefill_perf = 1 / (loop_start - start)
    decode_perf = (total_num_decoded_tokens) / (end - loop_start)
    total_perf = (total_num_decoded_tokens + decode_batch_size) / (end - start)

    if verbose:
        print()
        print("input=", prompt)
        print("output=", generated_texts)
        print("Prefill time a.k.a TTFT is=", round(prefill_perf, 2))
        print("Decode token/sec is=", round(decode_perf * batch_size, 2))
        print("Total token/sec is=", round(total_perf * batch_size, 2))
        print("Total (E2E) inference time is=", round(total_perf, 2))
        return
    print()
    print("===================== Performance Stats =====================")
    
    # if batch_size > 1:
    #     print("Prefill time a.k.a TTFT (batch) is :", round(prefill_perf, 2), "s")
    #     print("Decode (batch):", round(decode_perf * batch_size, 2), "tok/s")
    #     print("E2E (batch):", round(total_perf * batch_size, 2), "tok/s")
    #     print("Total (E2E) inference time (batch) is=", round(total_perf, 2), "s")
    # else:
    print("TTFT:", round(loop_start - start, 2), "s")
    print("E2ET:", round(end - start, 2), "s")
    print("Prefill:", round(prefill_perf, 2), "tok/s")
    print("Decode:", round(decode_perf, 2), "tok/s")
    print("E2E:", round(total_perf, 2), "tok/s")
    print("=============================================================")

    return total_num_decoded_tokens, prefill_perf, decode_perf, total_perf

def latency_stats_bertstyle(
        model_name: str,
        qpc: str,
        seq_len: int,
        prompt: str,
        device_id: List[int] = [0],
):
    session = QAICInferenceSession(qpc, device_id)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(prompt, return_tensors="np", max_length=seq_len, padding="max_length")
    next_token_id = inputs["input_ids"][0, -1]
    cur_len = inputs["attention_mask"].sum().item()
    print(prompt, end=" ", flush=True)
    init_len = cur_len
    start = perf_counter()
    while next_token_id != tokenizer.eos_token_id and cur_len <= seq_len:
        outputs = session.run(inputs)
        logits = outputs["logits"]
        next_token_id = logits[0, -1].argmax().item()
        inputs["input_ids"] = np.concatenate(
            [
                inputs["input_ids"][:, 1:],
                np.ones((1, 1), dtype=np.int64) * next_token_id,
            ],
            1,
        )
        inputs["attention_mask"] = np.concatenate([inputs["attention_mask"][:, 1:], np.ones((1, 1), dtype=np.int64)], 1)
        print(tokenizer.decode(next_token_id), end=" ", flush=True)
        cur_len += 1
    end = perf_counter()
    print()
    print(round((cur_len - init_len) / (end - start), 2), "tok/s")


def cloud_ai_100_exec_kv(
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        qpc: str,
        prompt: str,
        input_len: Optional[int] = None,
        generation_len: Optional[int] = None,
        device_id: List[int] = [0],
        enable_debug_logs: bool = False,
        stream: bool = False,
        write_io_dir: Optional[str] = None,
        automation: bool = False,
):
    if tokenizer.padding_side != "left":
        logger.warning(f"Please use padding_side='left' while initializing the tokenizer")
        tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Load QPC
    session = QAICInferenceSession(qpc, device_id, enable_debug_logs=enable_debug_logs)
    # Read prompt and ctx len from session
    prompt_len = max([x[session.binding_index_map["input_ids"]][1][1] for x in session.allowed_shapes])
    ctx_len = session.allowed_shapes[0][session.binding_index_map["attention_mask"]][1][1]
    if input_len is None:
        input_len = prompt_len
    if generation_len is None:
        generation_len = ctx_len
    num_chunks = -(input_len // -prompt_len)  # ceil divide without float
    input_len = num_chunks * prompt_len  # Convert input_len to a multiple of prompt_len
    assert input_len <= ctx_len, "input_len should be less than ctx_len"
    # Skip inputs/outputs
    session.skip_buffers([x for x in session.input_names if x.startswith("past_")])
    session.skip_buffers([x for x in session.output_names if x.endswith("_RetainedState")])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Prepare inputs for first iteration
    start = perf_counter()
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=input_len)
    batch_size = inputs["input_ids"].shape[0]
    inputs["position_ids"] = (np.cumsum(inputs["attention_mask"], 1) - 1) * inputs["attention_mask"]
    inputs["attention_mask"] = np.concatenate(
        [
            inputs["attention_mask"].astype(bool),
            np.zeros((batch_size, ctx_len - input_len), dtype=bool),
        ],
        1,
    )
    cache_index = np.array([0])
    inputs["cache_index"] = cache_index
    generated_ids = np.full((batch_size, generation_len - input_len + 1), tokenizer.pad_token_id)
    if stream:
        print(0, prompt[0], end=" ", flush=True)
    # Run prefill
    for i in range(num_chunks):
        chunk_inputs = inputs.copy()
        chunk_inputs["input_ids"] = inputs["input_ids"][:, i * prompt_len: (i + 1) * prompt_len]
        chunk_inputs["position_ids"] = inputs["position_ids"][:, i * prompt_len: (i + 1) * prompt_len]
        chunk_inputs["attention_mask"] = inputs["attention_mask"].copy()
        chunk_inputs["attention_mask"][:, (i + 1) * prompt_len:] = False
        outputs = session.run(chunk_inputs)
        if write_io_dir:
            write_io_files(inputs, outputs, write_io_dir, "prefill", "aic_batch_io", True, False)
        cache_index += prompt_len
    # Get first token
    logits = outputs["logits"]
    if len(logits.shape) == 2:
        logits = np.expand_dims(logits, 1)
    next_token_id = logits.argmax(2)
    inputs["input_ids"] = next_token_id
    inputs["position_ids"] = inputs.pop("attention_mask").sum(1, keepdims=True)
    generated_ids[:, cache_index[0] - input_len] = next_token_id.squeeze(1)
    if stream:
        print(tokenizer.decode(next_token_id[0]), end=" ", flush=True)
    # Skip attention_mask from next iteration to use retained attention_mask
    session.skip_buffers(["attention_mask"])
    loop_start = perf_counter()
    finished_sequences = next_token_id == tokenizer.eos_token_id
    while not finished_sequences.all() and cache_index[0] < generation_len:
        outputs = session.run(inputs)
        if write_io_dir:
            write_io_files(inputs, outputs, write_io_dir, "decode", "aic_batch_io", True, False)
            write_io_dir = None
        # Prepare inputs for next iteration
        logits = outputs["logits"]
        if len(logits.shape) == 2:
            logits = np.expand_dims(logits, 1)
        next_token_id = logits.argmax(2)
        finished_sequences |= next_token_id == tokenizer.eos_token_id
        inputs["input_ids"] = next_token_id
        inputs["position_ids"] += 1
        cache_index += 1
        generated_ids[:, cache_index[0] - input_len] = next_token_id.squeeze(1)
        if stream:
            print(tokenizer.decode(next_token_id[0]), end=" ", flush=True)
    end = perf_counter()
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    for i in range(1 if stream else 0, batch_size):
        print()
        print(i, prompt[i], generated_texts[i])
    prefill_time = loop_start - start
    decode_perf = (cache_index.item() - input_len - 1) / (end - loop_start)
    total_perf = (cache_index.item() - input_len) / (end - start)
    total_time = end - start
    print()
    if automation:
        print()
        print("input=", prompt)
        print("output=", generated_texts)
        print("Prefill time a.k.a TTFT is=", round(prefill_time, 2))
        print("Decode token/sec is=", round(decode_perf * batch_size, 2))
        print("Total token/sec is=", round(total_perf * batch_size, 2))
        print("Total (E2E) inference time is=", round(total_time, 2))
        return
    print()
    print("===================== Performance Stats =====================")
    if batch_size > 1:
        print("Prefill time a.k.a TTFT (batch) is :", round(prefill_time, 2), "s")
        print("Decode (batch):", round(decode_perf * batch_size, 2), "tok/s")
        print("E2E (batch):", round(total_perf * batch_size, 2), "tok/s")
        print("Total (E2E) inference time (batch) is=", round(total_time, 2), "s")
    else:
        print("Prefill time a.k.a TTFT is=", round(prefill_time, 2), "s")
        print("Decode:", round(decode_perf, 2), "tok/s")
        print("E2E:", round(total_perf, 2), "tok/s")
        print("Total (E2E) inference time is=", round(total_time, 2), "s")
    print("=============================================================")


def cloud_ai_100_exec_kv_cb(
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        qpc: str,
        prompt: str,
        input_len: Optional[int] = None,
        generation_len: Optional[int] = None,
        device_id: List[int] = [0],
        enable_debug_logs: bool = False,
        stream: bool = True,
        write_io_dir: Optional[str] = None,
        automation: bool = False,
        decode_batch_size: int = 4,
        prefill_batch_size: int = 1,
):
    # tokenizer = transformers.AutoTokenizer.from_pretrained("lu-vae/llama-68m-fft", padding_side="left", trust_remote_code=True)
   
    if tokenizer.padding_side != "left":
        logger.warning(f"Please use padding_side='left' while initializing the tokenizer")
        tokenizer.padding_side = "left"

    # Assign end of sentence token when pad_token_ID is none
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Fetch the vocab_size
    vocab_size= len(tokenizer)
    
    # Load QPC
    session = QAICInferenceSession(qpc, device_id, enable_debug_logs=enable_debug_logs)

    # Skip inputs/outputs
    session.skip_buffers([x for x in session.input_names if x.startswith("past_")])
    session.skip_buffers([x for x in session.output_names if x.endswith("_RetainedState")])
    
    # TODO need to read from the session bindings corresponding to prefill and decode according to stage
    batch_size, ctx_len = session.bindings[session.binding_index_map["attention_mask"]].dims
    print(f"session batch_size: {batch_size} , ctx_len: {ctx_len}")

    # Initiate a prefill queue. 
    prefill_queue = []

    # FIXME assumes prefill batch size is always 1
    set_logits_bsize(session, prefill_batch_size, vocab_size)

    # Read prompt, batch size and ctx len from session
    input_ids_index = session.binding_index_map["input_ids"]
    prompt_len = max([x[input_ids_index][1][1] for x in session.allowed_shapes]+
                     [session.bindings[input_ids_index].dims[1]])
    
    
    if input_len is None:
        input_len = prompt_len
    if generation_len is None:
        generation_len = ctx_len

    # Truncate prompts to required size
    if len(prompt) < decode_batch_size:
        print(f"Repeating prompt {decode_batch_size} times")
        prompt = prompt * -(decode_batch_size // -len(prompt))  # Repeat prompt to required size
    prompt = prompt[:decode_batch_size]  

    # FIXME for now, generating random numbers to decide how many tokens to generate for each prompt request
    req_max_length = list(np.random.randint(low=20, high=100, size=decode_batch_size))
    
    # add all prompts to the prefill queue
    prefill_queue = list(prompt)
    print("Request queue initially: ", prefill_queue)
    print("Max tokens/request:", req_max_length)

    # Initialize cache index and batch index. 
    cache_index = np.zeros((batch_size, 1), np.int64)
    batch_index = np.reshape(np.array(np.arange(batch_size), np.int64), (batch_size, 1))
    
    # initialize empty list to store generated tokens for each prompt
    generated_ids = [[] for _ in range(decode_batch_size)]
    # store the length of each prompt requested
    input_lengths = [0 for _ in range(decode_batch_size)]
    # store the number of prompts processed out of the prompt_queue
    num_prompts_processed = 0
    # initialize dynamic container which will hold all the global request ids (position in prompt request queue)
    # of the prompts currently being processed
    current_batch_req_ids = []
    
     # Prepare inputs for first iteration
    start = perf_counter()
    decode_inputs = create_decode_inputs(decode_batch_size, tokenizer, cache_index, batch_index, ctx_len)
    # print("####", decode_inputs)
    # iteratively run prefill with bs=1 for decode_batch_size number of requests to fill the decode bs=decode_batch_size -long container

    ##TODO handle when prefill queue size is less than the decode batch size
    # Iterate over the batch size
    for bi in range(decode_batch_size):
        prefill_queue, decode_inputs, current_batch_req_ids, input_lengths, num_prompts_processed, generated_ids = get_decode_input(
            bi, prefill_queue, session, tokenizer, prompt_len, ctx_len, decode_batch_size, decode_inputs,
            current_batch_req_ids, input_lengths, num_prompts_processed, generated_ids, write_io_dir)

    # FIXME currently we CANNOT Skip attention_mask from next iteration to use retained attention_mask
    # session.skip_buffers(["attention_mask"])
    # all_inputs.pop("attention_mask")
    print("** INITIAL PREFILL DONE **")
    
    # update logits placeholder for multi-batch decode
    set_logits_bsize(session, decode_batch_size, vocab_size)
    
    
    loop_start = perf_counter()
    next_token_id = decode_inputs["input_ids"]

    niter = 0
    current_decode_ongoing = True
    while (len(prefill_queue) > 0) or current_decode_ongoing:
        
        # Run Decode untill prefill queue is not empty or if decode is not complete. 
        outputs = session.run(decode_inputs)

        # get the next token id for each batch element
        logits = outputs["logits"]

        # Fetch the next tocken ID from the second last max logit probability. 
        if len(logits.shape) == 2: 
            logits = np.expand_dims(logits, 1)
        next_token_id = logits.argmax(2)

        current_decode_ongoing = False

        # Iterate over each batch in decode stage 
        # Check if any of the batch is hitting eos token, if not continue decoding. 
        for idx in range(decode_batch_size):
            # Check if any of the sequences have reached eos/max length
            pid = current_batch_req_ids[idx]
            if (len(generated_ids[pid]) < req_max_length[pid]) and generated_ids[pid][-1] != tokenizer.eos_token_id:
                
                generated_ids[pid].append(next_token_id[idx, 0])
                current_decode_ongoing = True


            # If any of the batch compete decode, further decode will stop and it will run prefill 
            if (next_token_id[idx, 0] == tokenizer.eos_token_id or len(generated_ids[pid]) >= req_max_length[pid]):
                if len(prefill_queue) == 0:
                    # print("prefill queue is empty. no slot replacement. accepting over-compute")
                    continue
                print(f"** running prefill on new request at prompt queue index = {num_prompts_processed} **")
                # FIXME assumes that prefill queue will always be popped from the front
                # stop this sequence and replace it with a new one from the prefill queue
                # run prefill on the new request
                (prefill_queue, decode_inputs, current_batch_req_ids, input_lengths, num_prompts_processed,
                 generated_ids) = get_decode_input(idx, prefill_queue, session, tokenizer, prompt_len, ctx_len,
                                                   decode_batch_size, decode_inputs, current_batch_req_ids,
                                                   input_lengths, num_prompts_processed, generated_ids, write_io_dir)
                # reset the logits placeholder back to decode mode
                set_logits_bsize(session, decode_batch_size, vocab_size)
            else:
                # Increment the cache index and position ids for this idx to continue decoding
                # update the attention mask
                req_id = pid
                
                decode_inputs = update_decode_inputs(decode_inputs, idx, input_lengths, req_id, generated_ids,
                                                     next_token_id)
        # print(f"Decoding batch ids: ", current_batch_req_ids)
        # print(f"Decode Iteration: {niter} next token per batch element: {next_token_id}")
        niter += 1

    
    end = perf_counter()

    print("Generated ID lengths: ", [len(o) for o in generated_ids])
    print("Max Tokens per Request assumed: ", req_max_length)

    # Decode the decoder output. 
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    print("##### num_prompts_processed",num_prompts_processed)
    # Print the output if the steam flag is true. 
    # for i in range(1 if stream else 0, num_prompts_processed):
    for i in range(num_prompts_processed):
        print()
        print("Generarted ID ",i)
        print("Prompt :", prompt[i])
        print("Generated tokens :", generated_texts[i])

    # Calculate the performance numbers. 
    total_num_decoded_tokens, prefill_perf, decode_perf, total_perf = calculate_latency(generated_ids, decode_batch_size, loop_start, start, end, automation)
