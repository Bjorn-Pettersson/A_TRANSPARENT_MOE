"""Quick unit test for SequenceMoE routing (double-softmax Top-K).
Creates a tiny GPT with SequenceMoE layers and runs a single forward
to validate shapes, losses and that routing decisions are exposed.
"""
import torch
from model import GPTConfig, GPT

def run_test():
    # tiny config to keep memory tiny and run on CPU by default
    cfg = GPTConfig(
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        n_expert=4,
        n_routed_expert=2,
        vocab_size=1000,
    )

    model = GPT(cfg)
    # run on cpu to avoid requiring a GPU in the test environment
    device = torch.device('cpu')
    model.to(device)
    model.train()

    batch_size = 3
    seq_len = 8
    # random integer tokens in vocab range
    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)

    logits, main_loss, aux_loss = model(x, x)

    print("logits.shape:", getattr(logits, 'shape', None))
    print("main_loss:", main_loss.item())
    print("aux_loss:", aux_loss.item())

    # Check routing info is present
    latest = getattr(model, 'latest_routing', None)
    print("latest_routing available:", latest is not None)
    if latest is not None:
        freqs = latest.get('layer_expert_freq', None)
        print("per-layer freq count:", len(freqs))
        for i, f in enumerate(freqs):
            print(f" layer {i}: ", f)

if __name__ == '__main__':
    run_test()
