# LC0 NLA

LC0 is among the best chess engines in the world. It's also pretty neural network heavy.
This repo tries to train a natural language autoencoder for it.

## The current plan
- Create warm start data using Gemma4-31B (currently here)
- Use this to distill a Qwen or some smaller 8B model that we can easily finetune
- Train an NLA on LC0
- Get good at chess?
