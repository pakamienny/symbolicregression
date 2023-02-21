import random, string, os
from pathlib import Path
import json
from symbolicregression.envs.graph import *
from symbolicregression.envs.rejector import *
from symbolicregression.envs.features import *
from symbolicregression.envs import *

def rstr(length: int = 6) -> str:
    return "".join(random.choice(string.ascii_letters) for _ in range(length))

def main(save_folder, max_expressions, params):
    env = build_env(params)
    env.set_rng(np.random.RandomState())

    fpath = save_folder / f"shard.{rstr(8)}.jsonl"
    os.makedirs(save_folder, exist_ok=True)
    file_handler = open(fpath, "w")
    print(f"======== Saving under {fpath} ========")

    n_expressions = 0
    while n_expressions<max_expressions:
        n_ops = np.random.randint(params.min_ops, params.max_ops)
        max_n_vars = np.random.randint(params.min_vars, params.max_vars+1)
        expr, (x, y) = env.get_sample(n_observations=params.n_max_observations, n_ops=n_ops, max_n_vars=max_n_vars)
        to_save = {"prefix": expr.prefix(), "x": x[0].tolist(), "y": y[0].tolist()} 
        file_handler.write(json.dumps(to_save) + "\n")
        if n_expressions<10: print(expr)
        n_expressions+=1

if __name__ == "__main__":
    from parsers import get_parser

    # generate parser / parse parameters
    parser = get_parser()
    params, extra = parser.parse_known_args()

    main(save_folder=Path(extra[0]), max_expressions=int(extra[1]), params=params)

"""
for i in {1..100}
do
    echo "python -m generate_expressions /checkpoint/pakamienny/sr_opensource/dataset/ 500000"
done | stool single generateWalks --ngpu 0 --ncpu 1 --mem 40 --partition learnlab
"""