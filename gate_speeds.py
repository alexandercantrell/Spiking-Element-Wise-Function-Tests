import os
import torch
from torch import nn
from tqdm.auto import tqdm
import timeit
import pandas as pd
from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sew_functions = {
    'ADD': lambda x,y: x+y,
    'AND': lambda x,y: x*y,
    'AND_RELU': lambda x,y: nn.functional.relu(x+y-1),
    'AND_COND': lambda x,y: (x+y>1.).to(x),
    'IAND': lambda x,y: x*(1.-y),
    'IAND_RELU': lambda x,y: nn.functional.relu(x-y),
    'IAND_COND': lambda x,y: (x-y>0.).to(x),
    'NAND': lambda x,y: 1-(x*y),
    'NAND_RELU': lambda x,y: 1-nn.functional.relu(x+y-1),
    'NAND_COND': lambda x,y: (x+y<=1.).to(x),
    'OR': lambda x,y: x+y-(x*y),
    'OR_COND': lambda x,y: (x+y>0.).to(x),
    'NOR': lambda x,y: 1-x-y+(x*y),
    'NOR_COND': lambda x,y: (x+y==0.).to(x),
    'XOR': lambda x,y: x+y-(2*x*y),
    'XOR_MOD': lambda x,y: (x+y)%2,
    'XOR_COND': lambda x,y: (x+y==1.).to(x),
    'XNOR': lambda x,y: 1-x-y+(2*x*y),
    'XNOR_MOD': lambda x,y: 1-((x+y)%2),
    'XNOR_COND': lambda x,y: (x+y!=1.).to(x)
}

def generate_activations(n=100,p=.5):
    return (torch.rand(n**2) <= p).to(torch.float32).reshape((n,n)).to(device)

def main():
    x = generate_activations(500)
    y = generate_activations(500)
    times = {}
    for cnf_name, cnf in tqdm(sew_functions.items()):
        times[cnf_name] = {'time':timeit.timeit(partial(cnf,x,y),number=1000)}
    df = pd.DataFrame.from_dict(times,orient='index')
    df.index.name='cnf'
    print(df)
    df.to_csv(f'{device}_times.csv',index=True)

if __name__ == "__main__":
    main()