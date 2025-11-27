import json
import os
import argparse
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor


def save_output(output, file_name):
    """
    Saves output data to a specified file in JSONL format.
    """
    with open(file_name, 'a', encoding='utf-8') as f:
        for record in output:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def load_file(file_name):
    """
    Loads JSONL lines from a file into a list of dictionaries.
    """
    if os.path.isfile(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f]
            return records, len(records)
    return [], 0

def writer(query):
    try:
        api_key = "qwen2.5-72b-instruct"
        base_url = "http://127.0.0.1:21001/v1"
        model = "/data/home/Yanchu/llm_repo/Qwen2.5-72B-Instruct"
        client = OpenAI(api_key=api_key, base_url=base_url)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": query},
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"error: {e}")
        raise
    
def process(id_query_map, out_file, num_workers=None):
    records, existing_count = load_file(out_file)
    cnt = existing_count
    contents, input_cnt = load_file(id_query_map)
    if num_workers is None:
        num_workers = 1
    with tqdm(total=input_cnt, initial=existing_count, desc=f"Processing {id_query_map.split('/')[-1]}") as pbar:
        with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as executor:
            futures = {}
            next_submit = existing_count
            next_write = existing_count
            while next_write < input_cnt:
                while next_submit < input_cnt and len(futures) < int(num_workers):
                    c = contents[next_submit]
                    q = c["query"]
                    f = executor.submit(writer, q)
                    futures[next_submit] = (f, c)
                    next_submit += 1
                f_c = futures.get(next_write)
                if f_c is None:
                    break
                f, c = f_c
                r = f.result()
                data = {"index": c["index"], "response": r}
                save_output([data], out_file)
                cnt += 1
                pbar.update(1)
                futures.pop(next_write)
                next_write += 1
    print(f"CNT: {cnt}")
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process lines from an input file.")
    parser.add_argument("--query_file", type=str, help="Path to the query file.")
    parser.add_argument("--output_file", type=str, help="Path to the output file.")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of concurrent workers.")

    args = parser.parse_args()

    process(args.query_file, args.output_file, args.num_workers)

