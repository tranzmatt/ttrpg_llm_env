import sys
from safetensors import safe_open
from safetensors.torch import save_file

def merge_safetensor_files(input_files, output_file="model.safetensors"):
    tensors = {}
    metadata = None
    for idx, file in enumerate(input_files):
        print(f"Merging: {file}")
        with safe_open(file, framework="pt") as sf_tsr:
            if metadata is None:
                metadata = sf_tsr.metadata()
            for layer in sf_tsr.keys():
                blk_tensor = sf_tsr.get_tensor(str(layer))
                tensors[str(layer)] = blk_tensor
    save_file(tensors, output_file, metadata if metadata else {})
    print(f"✓ Saved merged model to {output_file}")

if __name__ == "__main__":
    args = sys.argv[1:]
    if '-o' not in args or len(args) < 3:
        print("Usage: python merge_safetensors.py shard1 shard2 ... -o output_file")
        sys.exit(1)
    out_idx = args.index('-o')
    input_files = args[:out_idx]
    output_file = args[out_idx+1] if len(args) > out_idx+1 else None
    if not input_files or not output_file:
        print("Error: Missing input files or output file name.")
        sys.exit(1)
    print(f"The following shards/chunks will be merged: {input_files}")
    merge_safetensor_files(input_files, output_file)

