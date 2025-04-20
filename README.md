# TGN Recommender System

A Temporal Graph Network (TGN) based recommender system that leverages temporal user-item interactions for personalized recommendations.

## Overview

This project implements a Temporal Graph Network for recommender systems, which:

- Captures temporal dynamics in user-item interactions
- Uses memory modules to store and update user/item states
- Leverages graph attention mechanisms for embedding computation
- Supports various types of user-item interactions (views, add-to-cart, transactions)

## Dataset

The project uses a retail e-commerce dataset containing:

- User-item interactions (`events.csv`)
- Item properties (`item_properties_part1.csv`, `item_properties_part2.csv`)
- Category hierarchy (`category_tree.csv`)

Dataset statistics:

- ~2.7M interactions
- ~1.4M unique users
- ~235K unique items
- Multiple interaction types (view, addtocart, transaction)
- Temporal properties with Unix timestamps

## Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd TGN_Rec-1
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install torch torch-geometric numpy pandas
```

## Usage

### Data Preprocessing

1. Place your dataset files in the `new_data` directory:

   - `events.csv`
   - `item_properties_part1.csv`
   - `item_properties_part2.csv`
   - `category_tree.csv`

2. Run the preprocessing script:

```bash
python preprocess_retail.py
```

### Training the Model

Run the CPU version of TGN:

```bash
python main_cpu.py --data retail \
                   --use_memory \
                   --memory_updater gru \
                   --embedding_module graph_attention \
                   --prefix retail_run \
                   --n_epoch 2 \
                   --bs 32 \
                   --num_neg_eval 3 \
                   --n_degree 5 \
                   --num_candidates 2
```

### Command Line Arguments

- `--data`: Dataset name (default: 'retail')
- `--use_memory`: Enable memory module
- `--memory_updater`: Type of memory updater ('gru' or 'rnn')
- `--embedding_module`: Type of embedding module ('graph_attention', 'graph_ngcf', 'graph_sum', 'identity', 'time')
- `--n_epoch`: Number of training epochs
- `--bs`: Batch size
- `--num_neg_eval`: Number of negative samples for evaluation
- `--n_degree`: Number of temporal neighbors
- `--num_candidates`: Number of candidates per batch

## Model Architecture

The TGN model consists of:

1. Memory Module: Stores and updates user/item states
2. Message Function: Computes messages between nodes
3. Message Aggregator: Aggregates messages from neighbors
4. Embedding Module: Computes node embeddings using graph attention
5. Memory Updater: Updates node memory based on new interactions

## Performance

The model achieves:

- Training loss: ~1.35
- Validation NDCG@10: ~0.65
- Test NDCG@10: ~0.65

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```
[Add citation information]
```
