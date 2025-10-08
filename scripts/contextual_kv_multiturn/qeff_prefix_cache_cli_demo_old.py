#!/usr/bin/env python3
"""
QEfficient Prefix Cache CLI v3

Topic-based multi-turn conversation system with true prefix caching.
Each topic gets its own conversation thread with isolated KV cache.
"""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Code that triggers warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import warnings
import onnxscript

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning, module="onnxscript")
    # Import or call the specific code that triggers the warning
    
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*Conversion of an array with ndim > 0 to a scalar.*",
    category=DeprecationWarning
)



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
    batch_index: int                        # KV cache slot (0 or 1)
    conversation_history: str = ""          # Full conversation context
    turn_count: int = 0                     # Number of turns in this topic
    last_used: float = 0.0                  # For LRU replacement
    topic_name: str = ""                    # Auto-generated topic name
    context_length: int = 0                 # Current context length in tokens


@dataclass
class ConversationResponse:
    """Response from a conversation turn."""
    response: str
    topic_name: str
    batch_index: int
    turn_count: int
    cache_status: str  # "NEW_TOPIC", "CONTINUE_TOPIC", "RESET_TOPIC"
    context_length: int
    similarity_score: float = 0.0


class TopicBasedConversationManager:
    """Manages topic-based conversations with 2 isolated KV cache slots."""
    
    def __init__(self, similarity_threshold: float = 0.3, max_context_length: int = 800, verbose: bool = False):
        self.similarity_threshold = similarity_threshold
        self.max_context_length = max_context_length
        self.verbose = verbose
        
        # Initialize 2 topic slots
        self.topic_slots: Dict[int, TopicConversation] = {
            0: TopicConversation(batch_index=0),
            1: TopicConversation(batch_index=1)
        }
    
    def generate_topic_name(self, user_input: str) -> str:
        """Auto-generate topic name from user input."""
        # Keyword-based topic detection
        keywords = {
            'Cricket': ['cricket', 'virat', 'kohli', 'centuries', 'runs', 'batting', 'bowler', 'wickets'],
            'Books': ['harry potter', 'book', 'author', 'novel', 'series', 'wrote', 'chapter', 'story'],
        }
        
        input_lower = user_input.lower()
        for topic, words in keywords.items():
            if any(word in input_lower for word in words):
                return topic
        
        # Fallback: use first few meaningful words
        words = [w for w in user_input.split()[:3] if len(w) > 2]
        return ' '.join(words).title() if words else "General"
    
    def route_to_topic_slot(self, user_input: str) -> Tuple[int, bool, str, float]:
        """Route user input to appropriate topic slot using string matching."""
        if self.verbose:
            print(f"Analyzing topic for: '{user_input[:50]}{'...' if len(user_input) > 50 else ''}'")
        
        # Generate topic name for the input
        input_topic_name = self.generate_topic_name(user_input)
        
        # Check existing topic slots for matching topic names
        for slot_id in [0, 1]:
            slot = self.topic_slots[slot_id]
            if slot.topic_name and slot.topic_name == input_topic_name:
                # Found matching topic - reuse this slot
                slot.last_used = time.time()
                
                if self.verbose:
                    print(f"   TOPIC MATCH: Continuing {slot.topic_name}")
                
                return slot_id, True, f"Continuing {slot.topic_name}", 1.0
        
        # No matching topic found - need new topic slot
        available_slot = self.get_available_slot()
        self.initialize_new_topic(available_slot, input_topic_name)
        
        if self.verbose:
            print(f"   NEW TOPIC: Created {input_topic_name} in slot {available_slot}")
        
        return available_slot, False, f"Starting {input_topic_name}", -1.0
    
    def get_available_slot(self) -> int:
        """Get available slot using LRU if both occupied."""
        # Check for empty slot first
        for slot_id in [0, 1]:
            if not self.topic_slots[slot_id].topic_name:
                return slot_id
        
        # Both occupied, use LRU
        lru_slot = min([0, 1], key=lambda x: self.topic_slots[x].last_used)
        
        if self.verbose:
            old_topic = self.topic_slots[lru_slot].topic_name
            print(f"   LRU REPLACEMENT: Replacing {old_topic} in slot {lru_slot}")
        
        self.reset_topic_slot(lru_slot)
        return lru_slot
    
    def initialize_new_topic(self, slot_id: int, topic_name: str):
        """Initialize a new topic in the specified slot."""
        self.topic_slots[slot_id] = TopicConversation(
            batch_index=slot_id,
            conversation_history="",
            turn_count=0,
            last_used=time.time(),
            topic_name=topic_name,
            context_length=0
        )
    
    def reset_topic_slot(self, slot_id: int):
        """Reset a topic slot for reuse."""
        self.topic_slots[slot_id] = TopicConversation(batch_index=slot_id)
    
    def check_context_length_reset(self, slot_id: int, new_context: str) -> bool:
        """Check if context is too long and needs reset."""
        # Estimate token count (rough approximation)
        estimated_tokens = len(new_context.split()) * 1.3  # Rough token estimation
        
        if estimated_tokens > self.max_context_length:
            slot = self.topic_slots[slot_id]
            if self.verbose:
                print(f"   CONTEXT RESET: {slot.topic_name} context too long ({estimated_tokens:.0f} tokens)")
            
            # Keep topic info but reset conversation
            slot.conversation_history = ""
            slot.turn_count = 0
            slot.context_length = 0
            return True
        return False
    
    def get_stats(self) -> Dict:
        """Get conversation statistics."""
        active_topics = [slot for slot in self.topic_slots.values() if slot.topic_name]
        
        return {
            'active_topics': len(active_topics),
            'topics': {
                f"Slot {slot.batch_index}": {
                    'topic_name': slot.topic_name,
                    'turn_count': slot.turn_count,
                    'context_length': slot.context_length,
                    'last_used': slot.last_used
                }
                for slot in active_topics
            }
        }


class QEffLLMManager:
    """Manages QEfficient LLM with true prefix caching using TextGeneration API."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct", verbose: bool = False):
        self.model_name = model_name
        self.verbose = verbose
        self.llm_model = None
        self.tokenizer = None
        self.generator = None
        self.max_generation_length = 50
        
        self._load_models()
    
    def _load_models(self):
        """Load and compile the LLM model."""
        
        print(f"\n\nLoading Model : {self.model_name}...")
        
        try:
            # Load LLM model with continuous batching
            self.llm_model = QEFFAutoModelForCausalLM.from_pretrained(
                self.model_name, 
                continuous_batching=True
            )
            
            # Compile with 2 KV cache slots
            self.llm_model.compile(
                prefill_seq_len=128,
                ctx_len=1024,
                full_batch_size=2,
                kv_cache_batch_size=4,  # Only 2 slots for our 2 topics
                num_cores=14
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Initialize TextGeneration
            self.generator = TextGeneration(
                tokenizer=self.tokenizer,
                qpc_path=self.llm_model.qpc_path,
                full_batch_size=2,
                ctx_len=1024
            )
            
            if self.verbose:
                print("LLM model loaded and compiled successfully.")
                
        except Exception as e:
            print(f"Error loading LLM model: {e}")
            sys.exit(1)
    
    def generate_with_full_prefill(self, full_context: str, batch_index: int) -> Tuple[str, float]:
        """Generate response using full prefill with specific batch_index - Fixed implementation."""
        start_time = time.time()
        
        # try:
        if self.verbose:
            print(f"    Running full prefill with batch_index={batch_index}")
        
        # Use the manual decode pattern from the working test
        # Run prefill with specific decode_batch_id
        outputs, position_ids, gen_len = self.generator._qaic_model.run_prefill(
            full_context,
            generation_len=self.max_generation_length,
            decode_batch_id=np.array(batch_index, dtype=np.int64).reshape(1, 1)
        )
        
        # Manual decode setup - following the test pattern
        # For batch isolation, we need to set up the arrays correctly
        if batch_index == 0:
            decode_inputs = {
                "input_ids": np.array([[outputs["logits"].argmax(2)[0][0]], [0]]),
                "position_ids": np.array([[position_ids[0][0]], [-1]]),
                "batch_index": np.array([[0], [1]], dtype=np.int64),
            }
        else:  # batch_index == 1
            decode_inputs = {
                "input_ids": np.array([[0], [outputs["logits"].argmax(2)[0][0]]]),
                "position_ids": np.array([[-1], [position_ids[0][0]]]),
                "batch_index": np.array([[0], [1]], dtype=np.int64),
            }
        
        # Set logits placeholder for decode
        logits_out_placeholder = np.zeros(
            (
                self.generator._qaic_model.full_batch_size,
                self.generator._qaic_model._decode_seq_len,
                self.generator._qaic_model._vocab_size,
            ),
            dtype=np.float32,
        )
        self.generator._qaic_model._session.set_buffers({"logits": logits_out_placeholder})
        
        # Manual decode loop
        generation_outputs = []
        for i in range(min(gen_len, self.max_generation_length)):
            generation_outputs.append(decode_inputs["input_ids"])
            outputs = self.generator._qaic_model._session.run(decode_inputs)
            logits = outputs["logits"]
            if len(logits.shape) == 2:
                logits = np.expand_dims(logits, 1)
            next_token_id = logits.argmax(2)
            
            decode_inputs["input_ids"] = next_token_id
            decode_inputs["position_ids"][batch_index][0] += 1
            
            # Check for EOS token
            if next_token_id[batch_index][0] == self.tokenizer.eos_token_id:
                break
        
        # Extract generated tokens for this batch_index
        generated_tokens = [int(val[batch_index]) for val in generation_outputs]
        
        # Decode the generated text
        if generated_tokens:
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            response_text = "Generated response"
        
        prefill_time = time.time() - start_time
        
        if self.verbose:
            print(f"    Generated response in {prefill_time:.3f}s")
        
        return f"{response_text}", prefill_time
            
        # except Exception as e:
        #     if self.verbose:
        #         print(f"    Error during low-level generation: {e}")
        #         print(f"    Falling back to standard generation...")
            
        #     # Fallback to standard generation
        #     try:
        #         exec_info = self.generator.generate(
        #             prompt=[full_context], 
        #             generation_len=self.max_generation_length,
        #             stream=False
        #         )
                
        #         if hasattr(exec_info, 'generated_texts') and exec_info.generated_texts:
        #             generated_text = exec_info.generated_texts[0]
                    
        #             # Remove the original prompt from the response if it's included
        #             if generated_text.startswith(full_context):
        #                 generated_text = generated_text[len(full_context):].strip()
                    
        #             response_text = generated_text
        #         else:
        #             response_text = "Generated response from QEfficient LLM"
                
        #         prefill_time = time.time() - start_time
        #         response_text += f" [Standard generation - batch_index={batch_index}]"
                
        #         return response_text, prefill_time
                
        #     except Exception as e2:
        #         if self.verbose:
        #             print(f"    Standard generation also failed: {e2}")
                
        #         # Final fallback
        #         prefill_time = time.time() - start_time
        #         return f"Generated response for batch_index={batch_index} [Fallback due to errors]", prefill_time


class QEffPrefixCacheCLI:
    """Main CLI for topic-based multi-turn conversations with prefix caching."""
    
    def __init__(self, similarity_threshold: float = 0.3, max_context_length: int = 800, enable_cache: bool = True, verbose: bool = False):
        self.similarity_threshold = similarity_threshold
        self.max_context_length = max_context_length
        self.enable_cache = enable_cache
        self.verbose = verbose
        
        # Initialize managers
        self.conversation_manager = TopicBasedConversationManager(
            similarity_threshold, max_context_length, verbose
        )
        self.llm_manager = QEffLLMManager("meta-llama/Llama-3.2-1B-Instruct", verbose)
        
        # print("LLM model loaded and compiled successfully!")
        # print(f"Configuration: max_context={max_context_length}")
        # print("Topic detection: Keyword-based string matching")
    
    def process_conversation_turn(self, user_input: str) -> ConversationResponse:
        """Process a single conversation turn with topic-based routing."""
        
        if self.enable_cache:
            # WITH CACHING: Use topic-based routing
            slot_id, is_continuation, routing_info, similarity = self.conversation_manager.route_to_topic_slot(user_input)
            slot = self.conversation_manager.topic_slots[slot_id]
            
            # Build conversation context
            if is_continuation and slot.conversation_history:
                # Multi-turn: append to existing conversation
                full_context = slot.conversation_history + f"\nUser: {user_input}\nAssistant: "
                cache_status = "CONTINUE_TOPIC"
            else:
                # First turn or new topic: start fresh
                full_context = f"User: {user_input}\nAssistant: "
                cache_status = "NEW_TOPIC"
            
            # Check context length and reset if needed
            needs_reset = self.conversation_manager.check_context_length_reset(slot_id, full_context)
            if needs_reset:
                # Start fresh after reset
                full_context = f"User: {user_input}\nAssistant: "
                cache_status = "RESET_TOPIC"
            
            # Generate response using full prefill
            response, prefill_time = self.llm_manager.generate_with_full_prefill(full_context, slot_id)
            
            # Update conversation state
            slot.conversation_history = full_context + response
            slot.turn_count += 1
            slot.last_used = time.time()
            slot.context_length = len(slot.conversation_history.split()) * 1.3  # Rough token estimate
            
            return ConversationResponse(
                response=response,
                topic_name=slot.topic_name,
                batch_index=slot_id,
                turn_count=slot.turn_count,
                cache_status=cache_status,
                context_length=int(slot.context_length),
                similarity_score=similarity
            )
        
        else:
            # WITHOUT CACHING: Always use batch_index=0, no topic routing
            if self.verbose:
                print(f"CACHE DISABLED: Using batch_index=0 for all prompts")
            
            # Simple context - no conversation history
            full_context = f"User: {user_input}\nAssistant: "
            
            # Always use batch_index=0
            response, prefill_time = self.llm_manager.generate_with_full_prefill(full_context, 0)
            
            return ConversationResponse(
                response=response,
                topic_name="No-Cache",
                batch_index=0,
                turn_count=1,
                cache_status="NO_CACHE",
                context_length=len(full_context.split()),
                similarity_score=0.0
            )
    
    def interactive_mode(self):
        """Run interactive CLI mode."""
        # print("\nInteractive Mode - Topic-Based Conversations")
        # print("=" * 50)
        print("Type your questions")
        # print("Commands: 'quit' to exit, 'stats' for statistics, 'help' for help")
        # print("\nTry asking about different topics like:")
        # print("- Cricket: 'Tell me about Virat Kohli'")
        # print("- Books: 'Who wrote Harry Potter?'")
        # print("- Then switch between topics to see conversation isolation!")
        
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
                
                # Process the conversation turn
                response = self.process_conversation_turn(user_input)
                
                print(f"\nAssistant: {response.response}")
                
                # Always show basic cache info (not just in verbose mode)
                cache_icon = "[NEW]" if response.cache_status == "NEW_TOPIC" else "[CONT]" if response.cache_status == "CONTINUE_TOPIC" else "[CONT]"
                # print(f"    {cache_icon} {response.cache_status} | Slot {response.batch_index} | Turn {response.turn_count}")
                
                if self.verbose:
                    print(f"       Context: {response.context_length} tokens")
                    if response.similarity_score >= 0:
                        print(f"       Similarity: {response.similarity_score:.3f}")
                
            except KeyboardInterrupt:
                print("\n\nExiting!")
                break
            # except Exception as e:
            #     print(f"❌ Error: {e}")
    
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
                print(f"    └─ {response.cache_status} | Slot {response.batch_index} | Turn {response.turn_count}")
            
            results.append((prompt, response))
        
        # Show summary
        self.show_batch_summary(results)
    
    def show_batch_summary(self, results: List[Tuple]):
        """Show summary of batch processing results."""
        print(f"\nBatch Processing Summary:")
        print("=" * 40)
        
        # Group by topic
        topic_usage = {}
        for prompt, response in results:
            topic_key = f"{response.topic_name} (Slot {response.batch_index})"
            if topic_key not in topic_usage:
                topic_usage[topic_key] = []
            topic_usage[topic_key].append((prompt, response))
        
        for topic_key, prompts in topic_usage.items():
            print(f"\n{topic_key}: {len(prompts)} prompts")
            for prompt, response in prompts:
                status_icon = "[NEW]" if response.cache_status == "NEW_TOPIC" else "[CONT]"
                print(f"  {status_icon} {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
    
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
                print(f"    Last used: {time.time() - topic_data['last_used']:.1f}s ago")
    
    def show_help(self):
        """Show help information."""
        print("\nQEfficient Prefix Cache CLI v3 - Help")
        print("=" * 40)
        print("This CLI demonstrates topic-based multi-turn conversations with prefix caching.")
        print("\nHow it works:")
        print("• Each topic gets its own conversation thread (KV cache slot)")
        print("• Similar questions are grouped into the same topic")
        print("• You can switch between topics and conversations continue naturally")
        print("\nCommands:")
        print("  quit/exit/q  - Exit the program")
        print("  stats        - Show topic and conversation statistics")
        print("  help/?       - Show this help")
        print("\nExample conversation flow:")
        print("1. 'Tell me about Virat Kohli' → Creates Cricket topic")
        print("2. 'How many centuries has he scored?' → Continues Cricket topic")
        print("3. 'Who wrote Harry Potter?' → Creates Books topic")
        print("4. 'What's the last book?' → Continues Books topic")
        print("5. 'Did Virat break any records?' → Back to Cricket topic")


def main():
    parser = argparse.ArgumentParser(description="QEfficient Prefix Cache CLI v3 - Topic-Based Multi-Turn")
    parser.add_argument("--threshold", "-t", type=float, default=0.3,
                       help="Topic similarity threshold (default: 0.3)")
    parser.add_argument("--max-context", "-m", type=int, default=800,
                       help="Maximum context length before reset (default: 800)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--batch", "-b", nargs="+",
                       help="Run in batch mode with provided prompts")
    parser.add_argument("--demo", action="store_true",
                       help="Run demo with multi-topic conversation examples")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable topic-based caching (use batch_index=0 for all)")
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = QEffPrefixCacheCLI(
        similarity_threshold=args.threshold,
        max_context_length=args.max_context,
        enable_cache=not args.no_cache,
        verbose=args.verbose
    )
    
    if args.demo:
        # Run demo mode with topic switching
        demo_prompts = [
            "Tell me about Virat Kohli",
            "How many centuries has he scored?",
            "Who wrote Harry Potter?",
            "What's the last Harry Potter book?",
            "Did Virat Kohli break any batting records?",
            "How many Harry Potter movies were made?",
            "What is Virat's highest score?",
            "Which Harry Potter book is the longest?"
        ]
        cli.batch_mode(demo_prompts)
        
    elif args.batch:
        # Run batch mode
        cli.batch_mode(args.batch)
        
    else:
        # Run interactive mode
        cli.interactive_mode()


if __name__ == "__main__":
    main()
