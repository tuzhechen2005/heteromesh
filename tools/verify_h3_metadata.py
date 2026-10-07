"""Explicit opt-in metadata probe; never downloads tensor/weight files.

Run: python tools/verify_h3_metadata.py --download --cache-dir /path/to/cache
Then: python tools/verify_h3_metadata.py --cache-dir /path/to/cache
"""
import argparse
import json
from pathlib import Path
import urllib.request
from heteromesh.h3_metadata import MODEL_REVISION, inspect_official


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true',help='Fetch only two hash-pinned metadata JSON files')
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--end',type=int,default=2)
    args=parser.parse_args()
    sources={'config.json':'transformer/config.json','weight-index.json':'transformer/diffusion_pytorch_model.safetensors.index.json'}
    if args.download:
        downloaded={}
        for name,relative in sources.items():
            url=f'https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/{MODEL_REVISION}/{relative}'
            with urllib.request.urlopen(url,timeout=30) as response:
                raw=response.read(1024*1024+1)
            if len(raw)>1024*1024:raise ValueError('metadata exceeds 1 MiB')
            downloaded[name]=raw
        # Check both originals before publishing any cache file.
        inspect_official(downloaded['config.json'],downloaded['weight-index.json'],args.start,args.end)
        args.cache_dir.mkdir(parents=True,exist_ok=True)
        for name,raw in downloaded.items():(args.cache_dir/name).write_bytes(raw)
    result=inspect_official((args.cache_dir/'config.json').read_bytes(),(args.cache_dir/'weight-index.json').read_bytes(),args.start,args.end)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
