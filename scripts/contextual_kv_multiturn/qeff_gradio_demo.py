#!/usr/bin/env python3
"""
QEfficient Prefix Cache Gradio Demo

A simple, minimalistic Gradio interface for showcasing multi-turn conversations
with and without contextual KV cache.
"""

import gradio as gr
import time
from typing import List, Tuple, Dict
from qeff_prefix_cache_cli import QEffPrefixCacheCLI, ConversationResponse


class GradioQEffWrapper:
    """Wrapper for QEfficient CLI to work with Gradio interface."""
    
    def __init__(self):
        self.cli_with_cache = None
        self.cli_without_cache = None
        self.current_cache_mode = True
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize both cached and non-cached CLI instances."""
        print("Initializing models... This may take a moment.")
        self.cli_with_cache = QEffPrefixCacheCLI(
            similarity_threshold=0.3,
            max_context_length=800,
            enable_cache=True,
            verbose=False
        )
        self.cli_without_cache = QEffPrefixCacheCLI(
            similarity_threshold=0.3,
            max_context_length=800,
            enable_cache=False,
            verbose=False
        )
        print("Models initialized successfully!")
    
    def process_message(
        self, 
        message: str, 
        history: List[Tuple[str, str]], 
        use_cache: bool
    ) -> Tuple[List[Tuple[str, str]], str, str]:
        """
        Process a user message and return updated chat history and metrics.
        
        Returns:
            (updated_history, metrics_text, status_text)
        """
        if not message.strip():
            return history, "", ""
        
        # Select appropriate CLI based on cache mode
        cli = self.cli_with_cache if use_cache else self.cli_without_cache
        
        # Process the conversation turn
        response: ConversationResponse = cli.process_conversation_turn(message)
        
        # Format the assistant response with timing info
        assistant_message = self._format_response(response)
        
        # Update chat history
        history.append((message, assistant_message))
        
        # Generate metrics text
        metrics_text = self._format_metrics(response, use_cache)
        
        # Generate status text
        status_text = self._format_status(response, use_cache)
        
        return history, metrics_text, status_text
    
    def _format_response(self, response: ConversationResponse) -> str:
        """Format the assistant response with timing information."""
        # Main response text
        formatted = response.response
        
        # Add timing info as a small footer
        cache_emoji = "⚡" if response.cache_status == "INCREMENTAL_PREFILL" else "🆕"
        timing_info = (
            f"\n\n---\n"
            f"⏱️ **Prefill:** {response.prefill_time:.3f}s ({response.prefill_tokens} tokens) | "
            f"**Decode:** {response.decode_time:.3f}s | "
            f"{cache_emoji} **Cache:** {response.cache_status}"
        )
        
        return formatted + timing_info
    
    def _format_metrics(self, response: ConversationResponse, use_cache: bool) -> str:
        """Format metrics for display."""
        mode = "🔵 Cache Enabled" if use_cache else "⚪ Cache Disabled"
        
        metrics = f"""### Performance Metrics

**Mode:** {mode}

**Current Turn:** {response.turn_count}

**Topic:** {response.topic_name} (Slot {response.batch_index})

**Cache Status:** {response.cache_status}

**Context Length:** {response.context_length} tokens

**Timing:**
- Prefill: {response.prefill_time:.3f}s ({response.prefill_tokens} tokens)
- Decode: {response.decode_time:.3f}s
- Total: {response.total_time:.3f}s
"""
        return metrics
    
    def _format_status(self, response: ConversationResponse, use_cache: bool) -> str:
        """Format status message."""
        if not use_cache:
            return "ℹ️ Running in **No-Cache** mode - all context is re-processed each turn"
        
        if response.cache_status == "INCREMENTAL_PREFILL":
            return f"✅ **Incremental Update** - Only {response.prefill_tokens} new tokens processed (cache hit!)"
        elif response.cache_status == "NEW_TOPIC":
            return f"🆕 **New Topic** - Starting fresh conversation in slot {response.batch_index}"
        elif response.cache_status == "RESET_TOPIC":
            return f"🔄 **Context Reset** - Topic exceeded max length, restarting"
        else:
            return "ℹ️ Processing..."
    
    def clear_conversation(self, use_cache: bool) -> Tuple[List, str, str]:
        """Clear the conversation history."""
        # Reinitialize the appropriate CLI
        if use_cache:
            self.cli_with_cache = QEffPrefixCacheCLI(
                similarity_threshold=0.3,
                max_context_length=800,
                enable_cache=True,
                verbose=False
            )
        else:
            self.cli_without_cache = QEffPrefixCacheCLI(
                similarity_threshold=0.3,
                max_context_length=800,
                enable_cache=False,
                verbose=False
            )
        
        return [], "### Performance Metrics\n\n*Start a conversation to see metrics*", "ℹ️ Conversation cleared"


def create_gradio_interface():
    """Create and configure the Gradio interface."""
    
    wrapper = GradioQEffWrapper()
    
    # Custom CSS for a clean, minimalistic look
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .chat-message {
        font-size: 14px !important;
    }
    .metrics-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        font-size: 13px;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
    """
    
    with gr.Blocks(css=custom_css, title="QEfficient Multi-Turn Conversation Demo") as demo:
        gr.Markdown("""
        # 🚀 QEfficient Multi-Turn Conversation Demo
        
        Experience the power of **Contextual KV Cache** for faster multi-turn conversations.
        Toggle between cached and non-cached modes to see the performance difference!
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # Main chat interface
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_label=True,
                    elem_classes=["chat-message"]
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your message here... (e.g., 'Tell me about Virat Kohli')",
                        lines=2,
                        scale=4
                    )
                
                with gr.Row():
                    send_btn = gr.Button("Send 📤", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear Chat 🗑️", scale=1)
                
                # Status message
                status_msg = gr.Markdown(
                    "ℹ️ Ready to chat! Toggle cache mode and start a conversation.",
                    elem_classes=["status-box"]
                )
            
            with gr.Column(scale=1):
                # Cache toggle
                cache_toggle = gr.Checkbox(
                    label="Enable Contextual KV Cache",
                    value=True,
                    info="Toggle to compare performance"
                )
                
                gr.Markdown("---")
                
                # Metrics display
                metrics_display = gr.Markdown(
                    "### Performance Metrics\n\n*Start a conversation to see metrics*",
                    elem_classes=["metrics-box"]
                )
        
        # Info section
        with gr.Accordion("ℹ️ How It Works", open=False):
            gr.Markdown("""
            ### Contextual KV Cache Benefits:
            
            - **Incremental Prefill**: Only new tokens are processed, not the entire history
            - **Topic-Based Routing**: Each topic gets its own KV cache slot
            - **Faster Follow-ups**: ~50-70% faster for follow-up questions in the same topic
            
            ### Try These Examples:
            
            1. **Topic 1 - Cricket:**
               - "Tell me about Virat Kohli"
               - "How many centuries has he scored?" ⚡ (Fast with cache!)
               - "When did he become captain?" ⚡ (Fast with cache!)
            
            2. **Topic 2 - Books:**
               - "Who wrote Harry Potter?"
               - "What's the last book about?" ⚡ (Fast with cache!)
            
            3. **Back to Topic 1:**
               - "What's Virat's highest score?" ⚡ (Fast with cache!)
            """)
        
        # Event handlers
        def submit_message(message, history, use_cache):
            return wrapper.process_message(message, history, use_cache)
        
        def clear_chat(use_cache):
            return wrapper.clear_conversation(use_cache)
        
        # Send button click
        send_btn.click(
            fn=submit_message,
            inputs=[msg_input, chatbot, cache_toggle],
            outputs=[chatbot, metrics_display, status_msg]
        ).then(
            fn=lambda: "",  # Clear input box
            outputs=[msg_input]
        )
        
        # Enter key press
        msg_input.submit(
            fn=submit_message,
            inputs=[msg_input, chatbot, cache_toggle],
            outputs=[chatbot, metrics_display, status_msg]
        ).then(
            fn=lambda: "",  # Clear input box
            outputs=[msg_input]
        )
        
        # Clear button click
        clear_btn.click(
            fn=clear_chat,
            inputs=[cache_toggle],
            outputs=[chatbot, metrics_display, status_msg]
        )
    
    return demo


if __name__ == "__main__":
    print("Starting QEfficient Gradio Demo...")
    print("=" * 50)
    
    demo = create_gradio_interface()
    
    # Launch the demo
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
