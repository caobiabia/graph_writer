from transformers import AutoTokenizer
import torch, json, os, random, multiprocessing, argparse
from tqdm import tqdm
import numpy as np
import traceback

# Path or HF repo name for Qwen3-0.6B tokenizer/model
MAX_LENGTH = 32768
PROC_NUM = 32  # number of worker processes (adjust based on CPU cores)
TMP_DIR = "multiprocess_data_qwen"


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Pre-tokenize merged SFT data for Qwen3-0.6B")
    parser.add_argument("--input", default="data/qwen/graph_writer/merged.jsonl", help="Path to merged prompt-response jsonl file")
    parser.add_argument("--output_dir", default="data/qwen/graph_writer", help="Directory to save outputs (npy)")
    parser.add_argument("--tokenizer_path", required=True, help="Path or HF repo name for Qwen3-0.6B tokenizer/model")
    return parser.parse_args(args)


def build_sample(prompt: str, response: str, tokenizer, PAD_ID, EOS_ID, MAX_LENGTH):
    """Encode a single prompt/response pair.

    Prompt tokens are masked in labels (-100) so that only response contributes to the loss.
    Returns: (input_ids: torch.LongTensor, labels: torch.LongTensor)
    """
    # Compose chat messages according to Qwen template
    messages_pre = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ""},
    ]
    # IDs up to start of assistant response
    pre_ids = tokenizer.apply_chat_template(messages_pre, add_generation_prompt=False, return_tensors="pt")[0]
    start = pre_ids.size(0)

    messages_full = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    full_ids = tokenizer.apply_chat_template(messages_full, add_generation_prompt=False, return_tensors="pt")[0]

    # Append EOS if missing
    if full_ids[-1] != EOS_ID:
        full_ids = torch.cat([full_ids, torch.tensor([EOS_ID], dtype=torch.long)])

    labels = torch.full_like(full_ids, -100)
    labels[start:] = full_ids[start:]

    return full_ids, labels


def process_segment(lines, rank, tokenizer, PAD_ID, EOS_ID, MAX_LENGTH, TMP_DIR):
    try:
        inputs_mat = torch.full((len(lines), MAX_LENGTH), PAD_ID, dtype=torch.long)
        labels_mat = torch.full((len(lines), MAX_LENGTH), -100, dtype=torch.long)
        valid = 0
        for line in tqdm(lines, position=rank):
            obj = json.loads(line)
            inp_ids, lbls = build_sample(obj["prompt"], obj["response"], tokenizer, PAD_ID, EOS_ID, MAX_LENGTH)
            if inp_ids.size(0) > MAX_LENGTH:
                continue  # skip overly long
            inputs_mat[valid, :inp_ids.size(0)] = inp_ids
            labels_mat[valid, :lbls.size(0)] = lbls
            valid += 1
        inputs_mat = inputs_mat[:valid]
        labels_mat = labels_mat[:valid]
        torch.save(inputs_mat, os.path.join(TMP_DIR, f"inputs_{rank}.pt"))
        torch.save(labels_mat, os.path.join(TMP_DIR, f"labels_{rank}.pt"))
    except Exception:
        with open("pretokenize_qwen_error.txt", "a", encoding="utf-8") as f:
            traceback.print_exc(file=f)


def main():
    args = parse_args()

    # Initialize tokenizer (slow version preferred for chat template)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    # Ensure pad token exists for matrix padding
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    PAD_ID = tokenizer.pad_token_id
    EOS_ID = tokenizer.eos_token_id

    with open(args.input, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    random.shuffle(all_lines)
    total = len(all_lines)
    print("Total samples:", total)

    # Prepare directories
    if os.path.exists(TMP_DIR):
        import shutil
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Spawn pool
    proc_num = min(PROC_NUM, multiprocessing.cpu_count())
    seg_size = total // proc_num + 1
    pool = multiprocessing.Pool(proc_num)
    for i in range(proc_num):
        seg = all_lines[i * seg_size:(i + 1) * seg_size]
        pool.apply_async(process_segment, args=(seg, i, tokenizer, PAD_ID, EOS_ID, MAX_LENGTH, TMP_DIR))
    pool.close()
    pool.join()

    # Aggregate
    input_tensors = []
    label_tensors = []
    for i in range(proc_num):
        input_tensors.append(torch.load(os.path.join(TMP_DIR, f"inputs_{i}.pt")))
        label_tensors.append(torch.load(os.path.join(TMP_DIR, f"labels_{i}.pt")))
    inputs = torch.cat(input_tensors, dim=0)
    labels = torch.cat(label_tensors, dim=0)

    # Filter rows without any supervision
    mask = ~(labels == -100).all(dim=1)
    inputs = inputs[mask]
    labels = labels[mask]

    np.save(os.path.join(args.output_dir, "inputs.npy"), inputs.numpy().astype(np.int64))
    np.save(os.path.join(args.output_dir, "labels.npy"), labels.numpy().astype(np.int64))

    print("Saved inputs.npy and labels.npy with shapes", inputs.shape, labels.shape)


if __name__ == "__main__":
    main()