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

from collections import deque 
import numpy as np
import transformers
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.logging_utils import logger
def read_prompts_txt_file(self, prompts_txt_file_path: str):
        prompt = []
        with open(prompts_txt_file_path, "r") as file:
            for line in file:
                prompt.append(line.strip())
        return prompt


def check_batch_size_and_num_prompts( prompt, prompts_txt_file_path, batch_size) -> List[str]:
        assert (
            prompt is not None or prompts_txt_file_path is not None
        ), "Please pass atleast one argument either using --prompt or --prompts_txt_file_path"
        if prompts_txt_file_path is not None:
            if prompt is not None:
                logger.warning("Found inputs passed using txt file as well as CLI, taking inputs from given txt file")
            prompt = read_prompts_txt_file(prompts_txt_file_path)
        if isinstance(prompt, str):
            prompt = eval(prompt)
        num_prompts = len(prompt)
        if batch_size > 1:
            assert (
                batch_size == num_prompts
            ), f"Mismatch between number of prompts {num_prompts} and batch size {batch_size}; please pass correct input argument"
        return prompt

class TextGeneration:
    def __init__(self, 
                tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                qpc_path: str,
                prompt: List[str],
                full_batch_size: int = 1,
                input_len: Optional[int] = None,
                generation_len: Optional[int] = None,
                device_id: List[int] = [0],
                enable_debug_logs: bool = False,
                stream: bool = True,
                write_io_dir: Optional[str] = None,) -> None:
        
        self.io_files = []
        self.tokenizer=tokenizer
        self.prompt=prompt
        self.qpc_path=qpc_path
        self.device_id=device_id
        self.input_len=input_len
        self.generation_len=generation_len
        self.enable_debug_logs=enable_debug_logs
        self.stream=stream
        self.write_io_dir=write_io_dir,
        self.full_batch_size=full_batch_size
        
        # Load QPC
        self.session = QAICInferenceSession(qpc_path, device_id, enable_debug_logs=enable_debug_logs)
        
        # Get Vocab size 
        self.vocab_size = self.get_vocab_size()
        self.prefill_seq_len = self.get_prefill_seq_len() 


    # def __call__(self, ) -> :
        # pass

    def set_tokenizer_params(self):
        print(self.tokenizer)
        if self.tokenizer.padding_side != "right":
            logger.warning("Please use padding_side='right' while initializing the tokenizer")
            self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
    def write_io_files(self, 
        inputs: Dict[str, np.ndarray],
        outputs: Dict[str, np.ndarray],
        write_io_dir: str,
        write_io_subdir: str,
        write_io_name: str,
        include_dims: bool = False,
        reset: bool = False,
    ):
    
        if reset:
            self.io_files = []
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
        self.io_files.append(io)
        with open(f"{write_io_dir}/{write_io_name}.json", "w") as fp:
            json.dump({"IO-files": io_files}, fp, indent=True)

    def calculate_latency(self, generated_ids, decode_batch_size, loop_start, start, end, verbose=False, generated_texts=None,prompt=None, batch_size=1 ):
        """
        This method will calculate the latency metrics
        
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

    def latency_stats_bertstyle(self, 
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

    def get_compilation_batch_size(self, qpc_path: str):
        qpc_base_path = os.path.dirname(os.path.normpath(qpc_path))
        specialization_file_path = os.path.join(qpc_base_path, "specializations.json")
        logger.info(f"specialization_file_path : {specialization_file_path}")
        with open(specialization_file_path, "r") as file:
            data = json.load(file)
        compilation_batch_size = int(data["specializations"][0]["batch_size"])
        return compilation_batch_size

    def get_batch_size_ctx_len(self, ):
        batch_size, _, ctx_len, _ = self.session.bindings[self.session.binding_index_map["past_key.0"]].dims
        return batch_size, ctx_len

    def get_prefill_seq_len(self, ):
        prefill_seq_len = max(
            [x[self.session.binding_index_map["input_ids"]][1][1] for x in self.session.allowed_shapes]
            + [self.session.bindings[self.session.binding_index_map["input_ids"]].dims[1]])
        return prefill_seq_len

    def get_vocab_size(self,):
            return [x[self.session.binding_index_map["logits"]] for x in self.session.allowed_shapes][0][1][2]

    def prepare_decode_inputs(self, input_ids, position_id, full_batch_size, batch_index):
        """
        This function creates the decode inputs.

        Returns:
            dict: The decode inputs.
        """
        decode_inputs = {}
        decode_inputs["input_ids"] = input_ids
        decode_inputs["position_ids"] = position_id
        decode_inputs["batch_index"] = batch_index

        return decode_inputs

    def run_prefill(self, prompt, tokenizer, prefill_batch_size, batch_index, session, write_io_dir): 

        # Run prefill
        inputs = tokenizer(prompt, return_tensors="np", padding=True)
        position_ids = inputs["attention_mask"].sum(1, keepdims=True)
        padded_len = inputs["input_ids"].shape[1]
        num_chunks = -(padded_len // -self.prefill_seq_len)  # ceil divide without float
        padded_len = num_chunks * self.prefill_seq_len  # Convert to a multiple of prompt_len

        # Set the prefill logic buffer
        logits_out_placeholder = np.zeros((prefill_batch_size, 1, self.vocab_size), dtype=np.float32)
        session.set_buffers({"logits":logits_out_placeholder})
        
        inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
        inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
        inputs["batch_index"] = batch_index
        
        for i in range(num_chunks):
            chunk_inputs = inputs.copy()
            chunk_inputs["input_ids"] = inputs["input_ids"][:, i * self.prefill_seq_len : (i + 1) * self.prefill_seq_len]
            chunk_inputs["position_ids"] = inputs["position_ids"][:, i * self.prefill_seq_len : (i + 1) * self.prefill_seq_len]
            outputs = session.run(chunk_inputs)
            if write_io_dir:
                    self.write_io_files(inputs, outputs, write_io_dir, "prefill", "aic_batch_io", True, False)
        return outputs, position_ids

    def cloud_ai_100_exec_kv_helper(self, 
    prompt: List[str],
    full_batch_size: int = 1,
    prefill_batch_size: Optional[int]=1,
    input_len: Optional[int] = None,
    generation_len: Optional[int] = None,
    write_io_dir: Optional[str] = None,
):
        stream =  True
        prefill_batch_size = 1
        full_batch_size = 4 # TODO we should not be passing full batch size or prefill batch size to execution. this needs to be fetched from the QPC. 

        # set tokenizer params 
        self.set_tokenizer_params()
        
        # Skip inputs/outputs
        self.session.skip_buffers([x for x in self.session.input_names + self.session.output_names if x.startswith("past_")])

        # Read prompt and ctx len from session
        batch_size, ctx_len = self.get_batch_size_ctx_len()
        
        
        # Truncate prompts to required size
        # TODO check this can be done prior as a input processing module. 
        # TODO Need to handle for BS > 1 and decode batch size 1
        if len(prompt) < full_batch_size:
            # TODO add a warning saying input size and batch/decode batch size mismatch. Repeating the input prompts. 
            print(f"Repeating prompt {full_batch_size} times")
            prompt = prompt * -(full_batch_size // -len(prompt))  # Repeat prompt to required size
        
        prompt = prompt[:full_batch_size]
        prompt_queue= deque(prompt)
    

        # initialize np arrays for storing the prefill output for all the decode batch size. 
        batch_index = np.arange(full_batch_size).reshape(-1, 1)
        batch_index_prefill = np.arange(prefill_batch_size).reshape(-1, 1)
        generated_ids = np.full((len(prompt_queue), ctx_len), self.tokenizer.pad_token_id)
        decode_input_ids = np.zeros((full_batch_size, 1), np.int64)
        decode_pos_ids = np.zeros((full_batch_size, 1), np.int64)
        generation_len = np.zeros((full_batch_size, 1), np.int64)  
        
        # Prepare inputs for prefill with full batch dimensions
        start = perf_counter()
        
        # run prefill and accumulate results for all inputs in the queue. 
        for decode_batch_id in range(full_batch_size):
            if stream:
                streamer = transformers.TextStreamer(self.tokenizer)
                streamer.on_finalized_text(prompt[0] + " ")

            next_prompt = prompt_queue.popleft()
            
            # run prefill for num_chunks
            outputs, position_ids =  self.run_prefill(next_prompt, self.tokenizer, prefill_batch_size, batch_index_prefill, self.session, write_io_dir)
            
            logits = outputs["logits"]
            if len(logits.shape) == 2:
                logits = np.expand_dims(logits, 1)
        
            # Get output token
            next_token_id = logits.argmax(2)
            
            # Store the generated values.
            decode_input_ids[decode_batch_id] = next_token_id
            decode_pos_ids[decode_batch_id] = position_ids
            generated_ids[decode_batch_id, 0] = next_token_id.squeeze(1)
            generation_len[decode_batch_id] = ctx_len - position_ids+1
            
            print(f"Prompt : {prompt} batch_index: {decode_batch_id} prefill output id:{next_token_id[0]} token: {self.tokenizer.convert_ids_to_tokens(next_token_id[0])}")

        # Prepare decode inputs inputs. 
        decode_inputs = self.prepare_decode_inputs(decode_input_ids, decode_pos_ids, full_batch_size, batch_index)
        
        # Set logits placeholder for decode
        logits_out_placeholder = np.zeros((full_batch_size, 1, self.vocab_size), dtype=np.float32)
        self.session.set_buffers({"logits":logits_out_placeholder})
    
        # Generate flag for tracking progress for each batch ID 
        current_decode_ongoing = np.full((full_batch_size, 1), True)

        # Generate an array for maintaining the tokens generated in each batch ID
        # TODO validate if this can be replaced with generated_ids. Fetching the count might be slower compared to this. 
        generated_id_current_index = np.ones((full_batch_size, 1), np.int64)

        # Generate a batch ID map for mapping the batch ID if input > full_batch_size. 
        # This ID map will be used for storing all generated tokens
        batch_id_map = {i:i for i in range(full_batch_size)}
        decode_count = 0 # TODO check this can be achieved using generated_id_current_index. this would be needed for calculating the performance. 

        # Decode loop timer start 
        loop_start = perf_counter()

        # Decode loop
        while prompt_queue or current_decode_ongoing.any():
            decode_count+=1
            outputs = self.session.run(decode_inputs)
            
            # Prepare inputs for next iteration
            logits = outputs["logits"]
            if len(logits.shape) == 2:
                logits = np.expand_dims(logits, 1)
            next_token_id = logits.argmax(2)

            print(f"Decode Iteration: {decode_count} next token : {[self.tokenizer.convert_ids_to_tokens(next_token_id[i]) for i in range(batch_size)]}")
            
            for decode_batch_id in range(full_batch_size):
                if next_token_id[decode_batch_id] == self.tokenizer.eos_token_id or generated_id_current_index[decode_batch_id] >= generation_len[decode_batch_id]:
                    if prompt_queue:
                        # run prefill for next prompt input. 
                        outputs, position_ids =  self.run_prefill(prompt_queue.popleft(), self.tokenizer, prefill_batch_size, batch_index_prefill, self.session, write_io_dir)
                        
                        logits = outputs["logits"]
                        if len(logits.shape) == 2:
                            logits = np.expand_dims(logits, 1)
                        
                        # Get output token
                        token_id = logits.argmax(2)
                        decode_input_ids[decode_batch_id] = token_id
                        decode_pos_ids[decode_batch_id] = position_ids
                        generation_len[decode_batch_id] = ctx_len - position_ids+1
                        batch_id_map[decode_batch_id] = max(batch_id_map.values())+1
                        generated_ids[batch_id_map[decode_batch_id], 0] = token_id.squeeze(1)
                        generated_id_current_index[decode_batch_id] = 0
                        
                        self.session.set_buffers({"logits":logits_out_placeholder})
                                
                    else:
                        current_decode_ongoing[decode_batch_id] = False
                else:
                    # If the generated sequence is valid and within generation len prepare for next decode
                    decode_inputs["input_ids"][decode_batch_id] = next_token_id[decode_batch_id]
                    decode_inputs["position_ids"][decode_batch_id] += 1
                    generated_ids[batch_id_map[decode_batch_id],generated_id_current_index] = next_token_id[decode_batch_id]
                    
                    generated_id_current_index[decode_batch_id]+=1
        print("generated_ids",generated_ids.shape)

        end = perf_counter()
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for i in range(1 if self.stream else 0, batch_size):
            print()
            print(i, prompt[i], generated_texts[i])

    
        total_time = end - start
        verbose = False
        total_num_decoded_tokens, prefill_perf, decode_perf, total_perf = self.calculate_latency(generated_ids, full_batch_size, loop_start, start, end, verbose, generated_texts,prompt, batch_size)
        latency_stats = (generated_texts, prefill_perf, decode_perf, total_perf, total_time)
        return latency_stats




def print_latency_stats_kv(
    prompt, generated_texts, batch_size, prefill_time, decode_perf, total_perf, total_time, automation: bool = False
):
    if automation:
        print()
        print("input=", prompt)
        print("output=", generated_texts)
        print("Prefill time a.k.a TTFT is=", round(prefill_time * batch_size, 2))
        print("Decode token/sec is=", round(decode_perf * batch_size, 2))
        print("Total token/sec is=", round(total_perf * batch_size, 2))
        print("Total (E2E) inference time is=", round(total_time, 2))
        return
    print()

    print("===================== Performance Stats =====================")
    if batch_size > 1:
        print("Prefill time a.k.a TTFT (batch) is :", round(prefill_time * batch_size, 2), "s")
        print("Decode (batch):", round(decode_perf * batch_size, 2), "tok/s")
        print("E2E (batch):", round(total_perf * batch_size, 2), "tok/s")
        print("Total (E2E) inference time (batch) is=", round(total_time, 2), "s")
    else:
        print("Prefill time a.k.a TTFT is=", round(prefill_time, 2), "s")
        print("Decode:", round(decode_perf, 2), "tok/s")
        print("E2E:", round(total_perf, 2), "tok/s")
        print("Total (E2E) inference time is=", round(total_time, 2), "s")
    print("=============================================================")

def cloud_ai_100_exec_kv(
    batch_size,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    qpc_path: str,
    prompt: Optional[List[str]] = None,
    device_id: List[int] = [0],
    input_len: Optional[int] = None,
    generation_len: Optional[int] = None,
    enable_debug_logs: bool = False,
    stream: bool = True,
    write_io_dir: Optional[str] = None,
    automation=False,
    full_batch_size: int = 1,
):
    generate_text = TextGeneration(tokenizer=tokenizer,
                prompt=prompt,
                qpc_path=qpc_path,
                device_id=device_id,
                input_len=input_len,
                generation_len=generation_len,
                enable_debug_logs=enable_debug_logs,
                stream=stream,
                write_io_dir=write_io_dir,
                full_batch_size=full_batch_size)
    if batch_size == 1:
        prefill_time = []
        decode_perf = []
        total_perf = []
        total_time = []
        generated_texts = []
        for i in range(len(prompt)):
            latency_stats = generate_text.cloud_ai_100_exec_kv_helper(
                # tokenizer=tokenizer,
                prompt=[prompt[i]],
                # qpc=qpc_path,
                # device_id=device_id,
                input_len=input_len,
                generation_len=generation_len,
                # enable_debug_logs=enable_debug_logs,
                # stream=stream,
                write_io_dir=write_io_dir,
                full_batch_size=full_batch_size,
            )
            generated_texts.append(latency_stats[0])
            prefill_time.append(latency_stats[1])
            decode_perf.append(latency_stats[2])
            total_perf.append(latency_stats[3])
            total_time.append(latency_stats[4])

        prefill_time = np.average(prefill_time)
        decode_perf = np.average(decode_perf)
        total_perf = np.average(total_perf)
        total_time = np.average(total_time)

    else:
        latency_stats = generate_text.cloud_ai_100_exec_kv_helper(
            tokenizer=tokenizer,
            prompt=prompt,
            qpc=qpc_path,
            device_id=device_id,
            input_len=input_len,
            generation_len=generation_len,
            enable_debug_logs=enable_debug_logs,
            stream=stream,
            write_io_dir=write_io_dir,
        )
        generated_texts, prefill_time, decode_perf, total_perf, total_time = latency_stats

    print_latency_stats_kv(
        prompt,
        generated_texts,
        batch_size,
        prefill_time,
        decode_perf,
        total_perf,
        total_time,
        automation=automation,
    )
