# LC0 NLA

LC0 is among the best chess engines in the world. It's also pretty neural network heavy.
This repo tries to train a natural language autoencoder for it.

## The current plan
- Create warm start data using Gemma4-31B
- Use this to distill a Qwen-3.5 2B model
- Train an NLA on LC0 (currently here)
- Get good at chess?