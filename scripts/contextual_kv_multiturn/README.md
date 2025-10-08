# Contextual KV Cache for Multi-Turn Conversations

A demonstration of **incremental KV cache updates** for efficient multi-turn conversations using QEfficient on Cloud AI 100.

## What is Contextual KV Cache?

Traditional multi-turn conversations re-process the entire conversation history at each turn, which is inefficient. **Contextual KV Cache** solves this by:

- **Incremental Prefill**: Only new tokens are processed, not the entire history
- **Topic-Based Routing**: Each conversation topic gets its own isolated KV cache slot
- **Position Tracking**: Continues from the last position instead of restarting
- **Performance Boost**: ~50-70% faster for follow-up questions in the same topic

## Files

- `qeff_prefix_cache_cli.py` - Main CLI implementation with incremental caching
- `qeff_gradio_demo.py` - Interactive web UI demo using Gradio

## Quick Start

### Prerequisites

```bash
# Install Gradio (for web UI demo)
pip install gradio
```

### CLI Usage

#### Interactive Mode

```bash
python scripts/contextual_kv_multiturn/qeff_prefix_cache_cli.py
```

Then start chatting:
```
Who is Virat Kohli?
Who wrote Harry Potter book?
How many centuries has he scored?
What's the last book summary?
When did he became the indian cricket captain?
how did the last book end?
What is his highest score in an innings?
```

#### Batch Mode

```bash
python scripts/contextual_kv_multiturn/qeff_prefix_cache_cli.py --batch \
    "Tell me about Virat Kohli" \
    "How many centuries?" \
    "Who wrote Harry Potter?" \
    "What's the last book?"
```

#### Compare With/Without Cache

```bash
# With cache (default)
python scripts/contextual_kv_multiturn/qeff_prefix_cache_cli.py

# Without cache (for comparison)
python scripts/contextual_kv_multiturn/qeff_prefix_cache_cli.py --no-cache
```

### Gradio Web UI

Launch the interactive web interface:

```bash
python scripts/contextual_kv_multiturn/qeff_gradio_demo.py
```

Then open your browser to `http://localhost:7860`

**Features:**
- Toggle between cached and non-cached modes
- Real-time performance metrics
- Visual feedback on cache hits
- Side-by-side comparison

## CLI Commands

| Command | Description |
|---------|-------------|
| `quit` / `exit` / `q` | Exit the program |
| `stats` | Show conversation statistics |
| `help` / `?` | Show help information |

## Configuration Options

```bash
python scripts/contextual_kv_multiturn/qeff_prefix_cache_cli.py \
    --full-batch-size 2 \          # Number of parallel conversation slots
    --kv-cache-batch-size 4 \      # KV cache batch size
    --ctx-len 4096 \               # Maximum context length
    --gen-len 500 \                # Maximum generation length
    --threshold 0.3 \              # Topic similarity threshold
    --max-context 800 \            # Max tokens before context reset
    --verbose                      # Enable detailed logging
```

## How It Works

### 1. Topic Detection
The system automatically detects conversation topics using keyword matching:
- **Cricket**: virat, kohli, centuries, runs, batting, etc.
- **Books**: harry potter, author, novel, series, etc.
- **General**: Falls back to first few words

### 2. Cache Slot Management
- Each topic gets its own KV cache slot (default: 2 slots)
- LRU (Least Recently Used) replacement when all slots are occupied
- Automatic context reset when exceeding max length

### 3. Incremental Prefill
```
Turn 1: "Tell me about Virat Kohli"
  → Full prefill (NEW_TOPIC)

Turn 2: "How many centuries?"
  → Incremental prefill (only new tokens)
  → Reuses cached KV from Turn 1
  → ~50-70% faster!
```

## Example Conversation Flow

```bash
> Tell me about Virat Kohli
[Cricket] Virat Kohli is an Indian cricketer...
    [NEW] Slot 0 | Turn 1 | Prefilled 25 tokens

> How many centuries has he scored?
[Cricket] He has scored over 70 international centuries...
    [INCR] Slot 0 | Turn 2 | Prefilled 8 tokens  # Only 8 new tokens!

> Who wrote Harry Potter?
[Books] J.K. Rowling wrote the Harry Potter series...
    [NEW] Slot 1 | Turn 1 | Prefilled 18 tokens

> What's the last book about?
[Books] The last book, Harry Potter and the Deathly Hallows...
    [INCR] Slot 1 | Turn 2 | Prefilled 7 tokens  # Fast again!

> What's Virat's highest score?
[Cricket] His highest score in ODIs is 183...
    [INCR] Slot 0 | Turn 3 | Prefilled 6 tokens  # Back to Cricket topic!
```

## Performance Metrics

The system tracks and displays:
- **Prefill Time**: Time to process context tokens
- **Decode Time**: Time to generate response tokens
- **Total Time**: End-to-end latency
- **Cache Status**: NEW_TOPIC, INCREMENTAL_PREFILL, or RESET_TOPIC
- **Token Counts**: Prefill tokens vs decode tokens

## Troubleshooting

### Model Not Found
```bash
# Ensure QEfficient is properly installed
pip install --upgrade QEfficient
```

### Out of Memory
```bash
# Reduce context length or batch size
python qeff_prefix_cache_cli.py --ctx-len 2048 --full-batch-size 1
```

### Gradio Port Already in Use
```bash
# Change the port in qeff_gradio_demo.py
# Line: demo.launch(server_port=7860)  # Change to different port
```

## Notes

- The system uses `meta-llama/Llama-3.2-1B-Instruct` by default
- Requires Cloud AI 100 hardware for optimal performance
- Context is automatically reset when exceeding max length
- Topic detection can be customized by modifying keyword lists
