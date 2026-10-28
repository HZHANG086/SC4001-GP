"""Run on compute node with torch/transformers, no network or pretrained weights."""
import torch
from transformers import RobertaConfig
from sc4001_a.model import LinearRoberta

for method in ('M1', 'M2'):
    cfg = RobertaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=3,
                        num_attention_heads=4, intermediate_size=64,
                        max_position_embeddings=34)
    model = LinearRoberta(cfg)
    model.configure(method)
    model.train()
    x = torch.randint(4, 100, (4, 16))
    mask = torch.ones_like(x)
    y = torch.tensor([0, 1, 0, 1])
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    if method == 'M1':
        a = model(input_ids=x, attention_mask=mask).logits
        b = model(input_ids=x, attention_mask=mask).logits
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.01)
    model(input_ids=x, attention_mask=mask, labels=y).loss.backward()
    optimizer.step()
    changed = {n for n,p in model.named_parameters() if not torch.equal(p, initial[n])}
    assert 'classifier.weight' in changed
    for n,p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None and n not in changed, n
        if n.startswith('roberta.encoder.layer.'):
            idx = int(n.split('.')[3])
            assert p.requires_grad == (method == 'M2' and idx in (1, 2)), n
    assert not model.roberta.embeddings.training
    assert not model.roberta.encoder.layer[0].training
    assert model.roberta.encoder.layer[-1].training == (method == 'M2')
    print(method, 'gradient, frozen weights and dropout checks passed')
