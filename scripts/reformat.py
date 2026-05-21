import json
import argparse
from pathlib import Path
import chess

def format_board(fen: str) -> str:
    """
    Converts a FEN string to a 2D board string with spaces.
    Example output:
    r n b q k b n r
    p p p p p p p p
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    P P P P P P P P
    R N B Q K B N R
    """
    board = chess.Board(fen)
    return str(board)

def reformat_teacher_prompt(prompt: str) -> str:
    """
    Extracts the JSON from the teacher_prompt, replaces 'fen' with 'board_state',
    and reconstructs the prompt string.
    """
    prefix = "Evidence JSON:\n"
    idx = prompt.find(prefix)
    if idx == -1:
        return prompt # Return unmodified if prefix not found
    
    text_part = prompt[:idx + len(prefix)]
    json_part = prompt[idx + len(prefix):]
    
    try:
        evidence = json.loads(json_part)
        if "fen" in evidence:
            fen = evidence["fen"]
            evidence["board_state"] = format_board(fen)
            del evidence["fen"]
        # Re-dump with the same settings used in the original build script
        return text_part + json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception as e:
        print(f"Warning: Error parsing prompt JSON: {e}")
        return prompt

def main() -> None:
    parser = argparse.ArgumentParser(description="Reformat teacher prompts to use board states instead of FEN strings.")
    parser.add_argument("--input", type=Path, default=Path("../data/pretrain/shard-00000.jsonl"), help="Path to the input JSONL file")
    parser.add_argument("--output", type=Path, default=Path("../data/pretrain/shard-00000_reformatted.jsonl"), help="Path to the output JSONL file")
    parser.add_argument("--stats", type=Path, default=Path("../data/pretrain/stats-00000.json"), help="Path to the stats JSON file to update")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading from {args.input}")
    print(f"Writing to {args.output}")

    count = 0
    prompt_lengths = []

    with args.input.open("r", encoding="utf-8") as infile, args.output.open("w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping line due to invalid JSON: {exc}")
                continue

            if "teacher_prompt" in record:
                record["teacher_prompt"] = reformat_teacher_prompt(record["teacher_prompt"])
                prompt_lengths.append(len(record["teacher_prompt"]))

            outfile.write(json.dumps(record) + "\n")
            count += 1
            if count % 10000 == 0:
                print(f"Processed {count} records...")

    print(f"Done processing {count} records.")

    # Update stats
    if prompt_lengths and args.stats.exists():
        new_avg = sum(prompt_lengths) / len(prompt_lengths)
        print(f"Updating prompt_length_avg in {args.stats} to {new_avg}")
        
        try:
            with args.stats.open("r", encoding="utf-8") as f:
                stats_data = json.load(f)
            
            stats_data["prompt_length_avg"] = new_avg
            
            with args.stats.open("w", encoding="utf-8") as f:
                json.dump(stats_data, f, indent=2, sort_keys=True)
            print("Stats updated successfully.")
        except Exception as e:
            print(f"Error updating stats file: {e}")
    else:
        print("Stats file not found or no prompts processed, skipping stats update.")

if __name__ == "__main__":
    main()
