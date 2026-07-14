# LC0 NLA

[LC0](https://lczero.org/) is among the best chess engines in the world. It's also pretty neural network heavy.
This repo tries to train a [natural language autoencoder (NLA)](https://transformer-circuits.pub/2026/nla/) for it.

## Sample rollouts from the AV
https://github.com/user-attachments/assets/cdd731b6-a0d8-47c7-8010-6d52ee37b114

*Note: the NLA cannot see the FEN or the eval. These positions are fed into LC0, an activation is captured, and that activation is passed into the AV*

## The current plan
- Create warm start data using Gemma4-31B (done)
- Use this to distill a Qwen-3.5 2B model (done)
- Train an NLA on LC0 (done)
- Get good at chess (trying)

## How good is it?

FVE is ~0.5, which is _okay_ but it does hallucinate details pretty often.

The text gets the gist of who's winning, whether things are quiet or totally lost, and rough themes like king safety / material / which stage of the game it is more often than not. It still fails to do so a decent fraction of the time. It hallucinates a lot of the details though: specific moves, top lines, piece placement, basically a decent chunk of details.

Basically, this is NOT an analysis explanation for LC0. It is, I think, pretty cool and fun though. Remember, that this model _never_ even sees the chess board EVER. So it's going purely off of LC0's brain, which I think is pretty fun!

## Architecture

Two heads on Qwen 2B:

- **AV** (activation verbalizer): linear projector from Leela activations into Qwen embedding space, then generate a summary.
- **AR** (activation reconstructor): Qwen encodes the summary, then an MLP maps back to Leela activation dim.

Activations come from the last layer of Leela (512-d, prefix length 64). (probably suboptimal, might consider trying earlier layers later, hopefully it pushes up FVE!)

## Training
### Warm Start
I used vllm to generate ~200k summaries of positions from Gemma-4 31B (Gemma was able to see the board state, the eval, the top lines, and stuff like that). This was then used to "warm-start" the AV and AR by SFT'ing on this text. The initial FVE was ~0.15 (without any joint NLA training).

### Joint NLA
Run rollouts on the AV, reconstruct them with the AR. Backprop on the AR like normal (using MSE and cosine loss. This is a different choice I made from the Anthropic paper (they didn't use cosine, and they use a log that might not be necessary)). You can't just backprop on the AV though, because sampling isn't differentiable. So, you treat the reconstructed MSE + cosine value as a reward, and apply GRPO!

It ended with a ~0.5 FVE. There were a few notable issues with huge KL divergence spikes, and all-in-all there's probably things you could improve. I didn't have the compute to run an insane amount of runs, but most of the things I did try plateaud at this ~0.5 value. Feel free to contribute if you have any ideas / improvements!

## Repo layout

- `src/start_data/` — positions, LC0 UCI, teacher prompts, activation dumps
- `src/warm_start/` — AV / AR warm-start
- `src/nla/` — Joint NLA training
- `scripts/` — dataset builders, teacher fill, FVE
