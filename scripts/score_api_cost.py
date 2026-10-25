"""Sum recorded API tokens and price-equivalent USD cost using explicit rates."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--logs',type=Path,nargs='+',required=True,help='Explicit API JSONL logs; include only desired stages/attempts')
    p.add_argument('--input-usd-per-million',type=float,required=True)
    p.add_argument('--cached-input-usd-per-million',type=float,required=True)
    p.add_argument('--output-usd-per-million',type=float,required=True)
    p.add_argument('--total-scenes',type=int,default=150)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rates=[a.input_usd_per_million,a.cached_input_usd_per_million,a.output_usd_per_million]
    if min(rates)<0 or a.total_scenes<1:p.error('Rates must be nonnegative; total scenes positive')
    if a.output.exists():p.error('Choose a new output path')
    paths=[f.resolve() for f in a.logs]
    if len(set(paths))!=len(paths):p.error('Duplicate log files')
    total=dict(input_tokens=0,cached_input_tokens=0,output_tokens=0,calls=0);missing=0
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():continue
            row=json.loads(line);u=row.get('usage')
            if not u:missing+=1;continue
            inp=u.get('prompt_tokens',u.get('input_tokens'));out=u.get('completion_tokens',u.get('output_tokens'))
            cached=u.get('prompt_tokens_details',{}).get('cached_tokens',u.get('input_tokens_details',{}).get('cached_tokens',u.get('prompt_cache_hit_tokens',0)))
            if inp is None or out is None or not 0<=cached<=inp or out<0:raise ValueError('Invalid token usage')
            total['input_tokens']+=inp;total['cached_input_tokens']+=cached;total['output_tokens']+=out;total['calls']+=1
    if missing:raise ValueError(f'{missing} log rows have no token usage; cannot claim complete cost')
    usd=((total['input_tokens']-total['cached_input_tokens'])*rates[0]+total['cached_input_tokens']*rates[1]+total['output_tokens']*rates[2])/1e6
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(**total,usd_total=usd,usd_per_scene=usd/a.total_scenes,total_scenes=a.total_scenes,
        rates_usd_per_million=rates,logs=[str(p) for p in paths],definition='Price equivalent for explicitly selected logs; not a provider invoice. Include retries and stages consistently across methods.'),indent=2,allow_nan=False))


if __name__=='__main__':main()
