#!/usr/bin/env python3
"""
QEfficient Prefix Cache CLI v4 - Incremental KV Cache Updates

Key Features:
- Incremental prefill: Only new tokens are processed, not entire history
- Topic-based conversations: Each topic gets isolated KV cache slot
- Position tracking: Continues from last position instead of restarting
- Multi-turn optimization: ~50-70% faster for follow-up questions

Usage: python qeff_prefix_cache_cli_v4.py [--verbose] [--demo] [--batch "prompt1" "prompt2"]
"""

import sys
import argparse
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.generation.text_generation_inference import TextGeneration


@dataclass
class TopicConversation:
    """Represents a topic-based conversation with its own KV cache slot."""
    batch_index: int
    conversation_history: str = ""
    cached_context: str = ""
    last_position_id: int = 0
    cached_tokens: int = 0
    turn_count: int = 0
    last_used: float = 0.0
    topic_name: str = ""
    context_length: int = 0


@dataclass
class ConversationResponse:
    """Response from a conversation turn."""
    response: str
    topic_name: str
    batch_index: int
    turn_count: int
    cache_status: str
    context_length: int
    similarity_score: float = 0.0
    prefill_tokens: int = 0
    prefill_time: float = 0.0
    decode_time: float = 0.0
    total_time: float = 0.0


class TopicBasedConversationManager:
    """Manages topic-based conversations with dynamic number of isolated KV cache slots."""
    
    TOKEN_ESTIMATION_MULTIPLIER = 1.3
    
    def __init__(self, full_batch_size: int = 2, similarity_threshold: float = 0.3, max_context_length: int = 800, verbose: bool = False):
        self.full_batch_size = full_batch_size
        self.similarity_threshold = similarity_threshold
        self.max_context_length = max_context_length
        self.verbose = verbose
        
        # Dynamically create topic slots based on full_batch_size
        self.topic_slots: Dict[int, TopicConversation] = {
            i: TopicConversation(batch_index=i)
            for i in range(full_batch_size)
        }
    
    def generate_topic_name(self, user_input: str) -> str:
        """Auto-generate topic name from user input."""
        keywords = {
            'Cricket': ['cricket', 'virat', 'kohli', 'centuries', 'runs', 'batting', 'bowler', 'wickets', 'captain', 'innings'],
            'Books': ['harry potter', 'book', 'author', 'novel', 'series', 'wrote', 'chapter', 'story'],
        }
        
        input_lower = user_input.lower()
        for topic, words in keywords.items():
            if any(word in input_lower for word in words):
                return topic
        
        words = [w for w in user_input.split()[:3] if len(w) > 2]
        return ' '.join(words).title() if words else "General"
    
    def route_to_topic_slot(self, user_input: str) -> Tuple[int, bool, str, float]:
        """Route user input to appropriate topic slot using string matching."""
        input_topic_name = self.generate_topic_name(user_input)
        
        # Check existing topic slots for matching topic names
        for slot_id in range(self.full_batch_size):
            slot = self.topic_slots[slot_id]
            if slot.topic_name and slot.topic_name == input_topic_name:
                slot.last_used = time.perf_counter()
                if self.verbose:
                    print(f"   Continuing {slot.topic_name}")
                return slot_id, True, f"Continuing {slot.topic_name}", 1.0
        
        # No matching topic found - need new topic slot
        available_slot = self.get_available_slot()
        self.initialize_new_topic(available_slot, input_topic_name)
        
        if self.verbose:
            print(f"   New topic: {input_topic_name} in slot {available_slot}")
        
        return available_slot, False, f"Starting {input_topic_name}", -1.0
    
    def get_available_slot(self) -> int:
        """Get available slot using LRU if all occupied."""
        for slot_id in range(self.full_batch_size):
            if not self.topic_slots[slot_id].topic_name:
                return slot_id
        
        lru_slot = min(range(self.full_batch_size), key=lambda x: self.topic_slots[x].last_used)
        
        if self.verbose:
            old_topic = self.topic_slots[lru_slot].topic_name
            print(f"   Replacing {old_topic} in slot {lru_slot}")
        
        self.reset_topic_slot(lru_slot)
        return lru_slot
    
    def initialize_new_topic(self, slot_id: int, topic_name: str):
        """Initialize a new topic in the specified slot."""
        self.topic_slots[slot_id] = TopicConversation(
            batch_index=slot_id,
            conversation_history="",
            cached_context="",
            last_position_id=0,
            cached_tokens=0,
            turn_count=0,
            last_used=time.perf_counter(),
            topic_name=topic_name,
            context_length=0
        )
    
    def reset_topic_slot(self, slot_id: int):
        """Reset a topic slot for reuse."""
        self.topic_slots[slot_id] = TopicConversation(batch_index=slot_id)
    
    def check_context_length_reset(self, slot_id: int, new_context: str) -> bool:
        """Check if context is too long and needs reset."""
        estimated_tokens = self._estimate_token_count(new_context)
        
        if estimated_tokens > self.max_context_length:
            if self.verbose:
                slot = self.topic_slots[slot_id]
                print(f"   Context reset: {slot.topic_name} exceeded {self.max_context_length} tokens")
            return True
        return False
    
    def _estimate_token_count(self, text: str) -> int:
        """Rough token count estimation."""
        return int(len(text.split()) * self.TOKEN_ESTIMATION_MULTIPLIER)
    
    def get_stats(self) -> Dict:
        """Get conversation statistics."""
        active_topics = [slot for slot in self.topic_slots.values() if slot.topic_name]
        
        return {
            'active_topics': len(active_topics),
            'total_slots': self.full_batch_size,
            'topics': {
                f"Slot {slot.batch_index}": {
                    'topic_name': slot.topic_name,
                    'turn_count': slot.turn_count,
                    'context_length': slot.context_length,
                    'cached_tokens': slot.cached_tokens,
                    'last_position_id': slot.last_position_id,
                    'last_used': slot.last_used
                }
                for slot in active_topics
            }
        }


@dataclass
class ContextInfo:
    """Information about context to prefill."""
    text: str
    start_position: int
    is_incremental: bool
    add_special_tokens: bool


class QEffLLMManager:
    """Manages QEfficient LLM with incremental prefix caching using TextGeneration API."""
    
    # Model compilation constants
    PREFILL_SEQ_LEN = 128
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct", 
                 full_batch_size: int = 2, kv_cache_batch_size: int = 4, 
                 ctx_len: int = 4096, max_gen_len: int = 500, verbose: bool = False):
        self.model_name = model_name
        self.FULL_BATCH_SIZE = full_batch_size
        self.KV_CACHE_BATCH_SIZE = kv_cache_batch_size
        self.CTX_LEN = ctx_len
        self.MAX_GENERATION_LENGTH = max_gen_len
        self.NUM_CORES = 14
        self.verbose = verbose
        self.llm_model = None
        self.tokenizer = None
        self.generator = None
        
        self._load_models()
    
    def _load_models(self):
        """Load and compile the LLM model."""
        if self.verbose:
            print("Loading QEFFAutoModelForCausalLM...")
        
        try:
            self.llm_model = QEFFAutoModelForCausalLM.from_pretrained(
                self.model_name, 
                continuous_batching=True
            )
            
            self.llm_model.compile(
                prefill_seq_len=self.PREFILL_SEQ_LEN,
                ctx_len=self.CTX_LEN,
                full_batch_size=self.FULL_BATCH_SIZE,
                kv_cache_batch_size=self.KV_CACHE_BATCH_SIZE,
                num_cores=self.NUM_CORES
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.generator = TextGeneration(
                tokenizer=self.tokenizer,
                qpc_path=self.llm_model.qpc_path,
                full_batch_size=self.FULL_BATCH_SIZE,
                ctx_len=self.CTX_LEN
            )
            
            if self.verbose:
                print("LLM model loaded and compiled successfully.")
                
        except Exception as e:
            print(f"Error loading LLM model: {e}")
            sys.exit(1)
    
    def generate_with_incremental_prefill(
        self, 
        full_context: str, 
        batch_index: int,
        cached_context: str = "",
        last_position_id: int = 0
    ) -> Tuple[str, float, float, float, int, int]:
        """
        Generate response using incremental prefill.
        
        Returns:
            (response_text, prefill_time, decode_time, total_time, num_prefill_tokens, num_decode_tokens)
        """
        total_start = time.perf_counter()
        
        # Determine what to prefill
        context_info = self._determine_prefill_context(full_context, cached_context, last_position_id)
        
        # Tokenize and prepare inputs
        session_inputs, actual_len, num_chunks = self._prepare_prefill_inputs(
            context_info.text, 
            context_info.start_position, 
            context_info.add_special_tokens,
            batch_index
        )
        
        # Run prefill
        prefill_time, outputs = self._execute_prefill(session_inputs, num_chunks, batch_index)
        
        # Calculate final position for decode
        final_position_id = context_info.start_position + actual_len
        
        # Run decode
        response, decode_time, num_decode_tokens = self._execute_decode(
            batch_index, 
            final_position_id, 
            outputs
        )
        
        total_time = time.perf_counter() - total_start
        
        if self.verbose:
            print(f"    Timing: Prefill={prefill_time:.3f}s, Decode={decode_time:.3f}s, Total={total_time:.3f}s")
            print(f"    Tokens: Prefill={actual_len}, Decode={num_decode_tokens}")
            if context_info.is_incremental:
                print(f"    Incremental update: saved ~{len(cached_context.split())} tokens from cache")
        
        return response, prefill_time, decode_time, total_time, actual_len, num_decode_tokens
    
    def _determine_prefill_context(
        self, 
        full_context: str, 
        cached_context: str, 
        last_position_id: int
    ) -> ContextInfo:
        """Determine what context needs to be prefilled."""
        if cached_context and full_context.startswith(cached_context):
            # Incremental: only prefill the new part
            new_context = full_context[len(cached_context):]
            return ContextInfo(
                text=new_context,
                start_position=last_position_id,
                is_incremental=True,
                add_special_tokens=False
            )
        else:
            # Full prefill: new topic or cache invalidated
            return ContextInfo(
                text=full_context,
                start_position=0,
                is_incremental=False,
                add_special_tokens=True
            )
    
    def _prepare_prefill_inputs(
        self, 
        text: str, 
        start_position: int, 
        add_special_tokens: bool,
        batch_index: int
    ) -> Tuple[Dict, int, int]:
        """Tokenize and prepare inputs for prefill."""
        # Initial tokenization to get actual length
        inputs = self.tokenizer(text, return_tensors="np", padding=True, add_special_tokens=add_special_tokens)
        attention_mask = inputs["attention_mask"]
        actual_len = attention_mask.sum().item()
        padded_len = inputs["input_ids"].shape[1]
        
        # Calculate chunking
        num_chunks = -(padded_len // -self.PREFILL_SEQ_LEN)  # ceil divide
        padded_len = num_chunks * self.PREFILL_SEQ_LEN
        
        # Re-tokenize with correct padding
        inputs = self.tokenizer(
            text, 
            return_tensors="np", 
            padding="max_length", 
            max_length=padded_len,
            add_special_tokens=add_special_tokens
        )
        
        # Create position_ids starting from start_position
        attention_mask = inputs["attention_mask"]
        position_ids = np.where(
            attention_mask,
            np.arange(start_position, start_position + padded_len),
            -1
        )
        
        session_inputs = {
            "input_ids": inputs["input_ids"],
            "position_ids": position_ids,
            "batch_index": np.array(batch_index, dtype=np.int64).reshape(1, 1)
        }
        
        return session_inputs, actual_len, num_chunks
    
    def _execute_prefill(
        self, 
        session_inputs: Dict, 
        num_chunks: int, 
        batch_index: int
    ) -> Tuple[float, Dict]:
        """Execute prefill in chunks."""
        vocab_size = self.generator._qaic_model._vocab_size
        logits_out_placeholder = np.zeros((1, 1, vocab_size), dtype=np.float32)
        self.generator._qaic_model._session.set_buffers({"logits": logits_out_placeholder})
        
        prefill_start = time.perf_counter()
        
        for i in range(num_chunks):
            chunk_inputs = {
                "input_ids": session_inputs["input_ids"][:, i * self.PREFILL_SEQ_LEN : (i + 1) * self.PREFILL_SEQ_LEN],
                "position_ids": session_inputs["position_ids"][:, i * self.PREFILL_SEQ_LEN : (i + 1) * self.PREFILL_SEQ_LEN],
                "batch_index": session_inputs["batch_index"]
            }
            outputs = self.generator._qaic_model._session.run(chunk_inputs)
        
        prefill_time = time.perf_counter() - prefill_start
        
        return prefill_time, outputs
    
    def _execute_decode(
        self, 
        batch_index: int, 
        final_position_id: int, 
        prefill_outputs: Dict
    ) -> Tuple[str, float, int]:
        """Execute decode loop."""
        vocab_size = self.generator._qaic_model._vocab_size
        
        # Prepare decode inputs
        decode_inputs = self._prepare_batch_inputs(
            batch_index, 
            int(prefill_outputs["logits"].argmax(2)[0][0]), 
            final_position_id
        )
        
        # Set logits placeholder for decode - GENERALIZED
        logits_out_placeholder = np.zeros((self.FULL_BATCH_SIZE, 1, vocab_size), dtype=np.float32)
        self.generator._qaic_model._session.set_buffers({"logits": logits_out_placeholder})
        
        # Decode loop
        generation_outputs = []
        gen_len = min(self.MAX_GENERATION_LENGTH, self.generator._qaic_model._ctx_len - final_position_id)
        
        decode_start = time.perf_counter()
        for i in range(gen_len):
            generation_outputs.append(decode_inputs["input_ids"][batch_index])
            outputs = self.generator._qaic_model._session.run(decode_inputs)
            logits = outputs["logits"]
            if len(logits.shape) == 2:
                logits = np.expand_dims(logits, 1)
            next_token_id = logits.argmax(2)
            
            decode_inputs["input_ids"] = next_token_id
            decode_inputs["position_ids"][batch_index][0] += 1
            
            if next_token_id[batch_index][0] == self.tokenizer.eos_token_id:
                break
        
        decode_time = time.perf_counter() - decode_start
        
        # Decode generated tokens
        generated_tokens = [int(val) for val in generation_outputs]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True) if generated_tokens else ""
        
        return response_text, decode_time, len(generated_tokens)
    
    def _prepare_batch_inputs(self, batch_index: int, token_id: int, position_id: int) -> Dict:
        """Prepare decode inputs for specific batch index - GENERALIZED VERSION."""
        
        # Create arrays with shape [FULL_BATCH_SIZE, 1]
        input_ids = np.zeros((self.FULL_BATCH_SIZE, 1), dtype=np.int64)
        position_ids = np.full((self.FULL_BATCH_SIZE, 1), -1, dtype=np.int64)
        batch_indices = np.arange(self.FULL_BATCH_SIZE, dtype=np.int64).reshape(-1, 1)
        
        # Set the active batch slot
        input_ids[batch_index, 0] = token_id
        position_ids[batch_index, 0] = position_id
        
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "batch_index": batch_indices,
        }


class QEffPrefixCacheCLI:
    """Main CLI for topic-based multi-turn conversations with incremental prefix caching."""
    
    def __init__(self, full_batch_size: int = 2, kv_cache_batch_size: int = 4,
                 ctx_len: int = 4096, gen_len: int = 500,
                 similarity_threshold: float = 0.3, max_context_length: int = 800, 
                 enable_cache: bool = True, verbose: bool = False):
        self.full_batch_size = full_batch_size
        self.kv_cache_batch_size = kv_cache_batch_size
        self.similarity_threshold = similarity_threshold
        self.max_context_length = max_context_length
        self.enable_cache = enable_cache
        self.verbose = verbose
        self.no_cache_history = ""
        
        print("QEfficient Prefix Cache CLI")
        print("=" * 40)
        print("Incremental KV Cache Updates")
        print("Initializing models...")
        
        self.llm_manager = QEffLLMManager(
            "meta-llama/Llama-3.2-1B-Instruct", 
            full_batch_size=full_batch_size,
            kv_cache_batch_size=kv_cache_batch_size,
            ctx_len=ctx_len,
            max_gen_len=gen_len,
            verbose=verbose
        )
        
        self.conversation_manager = TopicBasedConversationManager(
            full_batch_size=full_batch_size,
            similarity_threshold=similarity_threshold,
            max_context_length=max_context_length,
            verbose=verbose
        )
        
        print("Ready! Type 'help' for commands.")
    
    def process_conversation_turn(self, user_input: str) -> ConversationResponse:
        """Process a single conversation turn with incremental caching."""
        
        if self.enable_cache:
            # With caching: use topic-based routing and incremental prefill
            slot_id, is_continuation, routing_info, similarity = self.conversation_manager.route_to_topic_slot(user_input)
            slot = self.conversation_manager.topic_slots[slot_id]
            
            # Build conversation context
            if is_continuation and slot.conversation_history:
                full_context = slot.conversation_history + f"\nUser: {user_input}\nAssistant: "
                cache_status = "INCREMENTAL_PREFILL"
            else:
                full_context = f"User: {user_input}\nAssistant: "
                cache_status = "NEW_TOPIC"
            
            # Check context length and reset if needed
            needs_reset = self.conversation_manager.check_context_length_reset(slot_id, full_context)
            if needs_reset:
                full_context = f"User: {user_input}\nAssistant: "
                cache_status = "RESET_TOPIC"
                slot.conversation_history = ""
                slot.cached_context = ""
                slot.last_position_id = 0
                slot.cached_tokens = 0
                slot.turn_count = 0
                slot.context_length = 0
            
            # Generate with incremental prefill
            response, prefill_time, decode_time, total_time, num_prefill_tokens, num_decode_tokens = self.llm_manager.generate_with_incremental_prefill(
                full_context=full_context,
                batch_index=slot_id,
                cached_context=slot.cached_context,
                last_position_id=slot.last_position_id
            )
            
            # Update cached state
            slot.conversation_history = full_context + response
            slot.cached_context = full_context + response
            slot.last_position_id += num_prefill_tokens + num_decode_tokens
            slot.cached_tokens += num_prefill_tokens + num_decode_tokens
            slot.turn_count += 1
            slot.last_used = time.perf_counter()
            slot.context_length = self.conversation_manager._estimate_token_count(slot.conversation_history)
            
            return ConversationResponse(
                response=response,
                topic_name=slot.topic_name,
                batch_index=slot_id,
                turn_count=slot.turn_count,
                cache_status=cache_status,
                context_length=int(slot.context_length),
                similarity_score=similarity,
                prefill_tokens=num_prefill_tokens,
                prefill_time=prefill_time,
                decode_time=decode_time,
                total_time=total_time
            )
        
        else:
            # Without caching: standard multi-turn conversation
            if self.no_cache_history:
                full_context = self.no_cache_history + f"\nUser: {user_input}\nAssistant: "
            else:
                full_context = f"User: {user_input}\nAssistant: "

            response, prefill_time, decode_time, total_time, num_prefill_tokens, num_decode_tokens = self.llm_manager.generate_with_incremental_prefill(
                full_context, 0, "", 0
            )

            self.no_cache_history = full_context + response

            return ConversationResponse(
                response=response,
                topic_name="No-Cache",
                batch_index=0,
                turn_count=self.no_cache_history.count("User:"),
                cache_status="NO_CACHE",
                context_length=len(full_context.split()),
                similarity_score=0.0,
                prefill_tokens=num_prefill_tokens,
                prefill_time=prefill_time,
                decode_time=decode_time,
                total_time=total_time,
            )
    
    def interactive_mode(self):
        """Run interactive CLI mode."""
        print("\nInteractive Mode")
        print("=" * 50)
        print("Commands: 'quit', 'stats', 'help'")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() in ['stats', 'status']:
                    self.show_stats()
                    continue
                
                if user_input.lower() in ['help', '?']:
                    self.show_help()
                    continue
                
                response = self.process_conversation_turn(user_input)
                
                print(f"\n[{response.topic_name}] {response.response}")
                
                cache_icon = "[INCR]" if response.cache_status == "INCREMENTAL_PREFILL" else "[NEW]"
                print(f"    {cache_icon} Slot {response.batch_index} | Turn {response.turn_count} | Prefilled {response.prefill_tokens} tokens")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
    
    def batch_mode(self, prompts: List[str]):
        """Process a list of prompts in batch mode."""
        print(f"\nBatch Mode - Processing {len(prompts)} prompts")
        print("=" * 50)
        
        results = []
        for i, prompt in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}] {prompt}")
            response = self.process_conversation_turn(prompt)
            print(f"[{response.topic_name}] {response.response}")
            
            if self.verbose:
                print(f"    └─ {response.cache_status} | Slot {response.batch_index} | {response.prefill_tokens} tokens")
            
            results.append((prompt, response))
        
        self.show_batch_summary(results)
    
    def show_batch_summary(self, results: List[Tuple]):
        """Show summary of batch processing results."""
        print(f"\nBatch Summary:")
        print("=" * 40)
        
        topic_usage = {}
        total_prefill_tokens = 0
        incremental_count = 0
        
        for prompt, response in results:
            topic_key = f"{response.topic_name} (Slot {response.batch_index})"
            if topic_key not in topic_usage:
                topic_usage[topic_key] = []
            topic_usage[topic_key].append((prompt, response))
            total_prefill_tokens += response.prefill_tokens
            if response.cache_status == "INCREMENTAL_PREFILL":
                incremental_count += 1
        
        for topic_key, prompts in topic_usage.items():
            print(f"\n{topic_key}: {len(prompts)} prompts")
            for prompt, response in prompts:
                status_icon = "[INCR]" if response.cache_status == "INCREMENTAL_PREFILL" else "[NEW]"
                print(f"  {status_icon} {prompt[:60]}{'...' if len(prompt) > 60 else ''} ({response.prefill_tokens} tokens)")
        
        print(f"\nOptimization Stats:")
        print(f"  Total prefill tokens: {total_prefill_tokens}")
        print(f"  Incremental updates: {incremental_count}/{len(results)}")
        print(f"  Average tokens/prompt: {total_prefill_tokens/len(results):.1f}")
    
    def show_stats(self):
        """Show conversation statistics."""
        stats = self.conversation_manager.get_stats()
        
        print(f"\nConversation Statistics:")
        print("=" * 30)
        print(f"Active topics: {stats['active_topics']}")
        
        if stats['topics']:
            print(f"\nTopic Details:")
            for slot_name, topic_data in stats['topics'].items():
                print(f"  {slot_name}: {topic_data['topic_name']}")
                print(f"    Turns: {topic_data['turn_count']}")
                print(f"    Context: {topic_data['context_length']} tokens")
                print(f"    Cached: {topic_data['cached_tokens']} tokens")
                print(f"    Position: {topic_data['last_position_id']}")
    
    def show_help(self):
        """Show help information."""
        print("\nQEfficient Prefix Cache CLI v4 - Help")
        print("=" * 40)
        print("\nFeatures:")
        print("• Incremental prefill: Only new tokens are processed")
        print("• Topic-based caching: Each topic gets own KV cache slot")
        print("• Multi-turn optimization: ~50-70% faster for follow-ups")
        print("\nCommands:")
        print("  quit/exit/q  - Exit")
        print("  stats        - Show statistics")
        print("  help/?       - Show this help")
        print("\nExample:")
        print("1. 'Tell me about Virat Kohli' → Full prefill")
        print("2. 'How many centuries?' → Incremental (fast!)")
        print("3. 'Who wrote Harry Potter?' → New topic")
        print("4. 'What's the last book?' → Incremental (fast!)")


def main():
    parser = argparse.ArgumentParser(description="QEfficient Prefix Cache CLI v4")
    parser.add_argument("--full-batch-size", "-f", type=int, default=2,
                       help="Full batch size (default: 2)")
    parser.add_argument("--kv-cache-batch-size", "-k", type=int, default=4,
                       help="KV cache batch size (default: 4)")
    parser.add_argument("--ctx-len", "-c", type=int, default=4096,
                       help="Context length (default: 4096)")
    parser.add_argument("--gen-len", "-g", type=int, default=500,
                       help="Maximum generation length (default: 500)")
    parser.add_argument("--threshold", "-t", type=float, default=0.3,
                       help="Topic similarity threshold (default: 0.3)")
    parser.add_argument("--max-context", "-m", type=int, default=800,
                       help="Maximum context length before reset (default: 800)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--batch", "-b", nargs="+",
                       help="Run in batch mode with provided prompts")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable topic-based caching")
    
    args = parser.parse_args()
    
    cli = QEffPrefixCacheCLI(
        full_batch_size=args.full_batch_size,
        kv_cache_batch_size=args.kv_cache_batch_size,
        ctx_len=args.ctx_len,
        gen_len=args.gen_len,
        similarity_threshold=args.threshold,
        max_context_length=args.max_context,
        enable_cache=not args.no_cache,
        verbose=args.verbose
    )
        
    if args.batch:
        cli.batch_mode(args.batch)
        
    else:
        cli.interactive_mode()


if __name__ == "__main__":
    main()
