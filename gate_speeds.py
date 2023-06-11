import os
import torch
from torch import nn
from tqdm.auto import tqdm
import timeit
import pandas as pd
from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def and_where(x,y):
    z = torch.add(x,y)
    return torch.where(z>1.,1.,0.)

def iand_where(x,y):
    z = torch.sub(x,y)
    return torch.where(z<0.,0.,z)

def nand_where(x,y):
    return 1-and_where(x,y)

def or_where(x,y):
    z = torch.add(x,y)
    return torch.where(z>1.,1.,z)

def nor_where(x,y):
    return 1-or_where(x,y)

def xor_where(x,y):
    z = torch.add(x,y)
    return torch.where(z>1.,0.,z)

def xnor_where(x,y):
    return 1-xor_where(x,y)


sew_functions = {
    'ADD_DEFAULT': lambda x,y: x+y,
    'AND_DEFAULT': lambda x,y: x*y,
    'AND_RELU': lambda x,y: nn.functional.relu(x+y-1),
    'AND_COND': lambda x,y: (x+y>1.).to(x),
    'AND_WHERE': and_where,
    'IAND_DEFAULT': lambda x,y: x*(1.-y),
    'IAND_RELU': lambda x,y: nn.functional.relu(x-y),
    'IAND_COND': lambda x,y: (x-y>0.).to(x),
    'IAND_WHERE': iand_where,
    'NAND_DEFAULT': lambda x,y: 1-(x*y),
    'NAND_DEFAULT+': lambda x,y: ((x*y)-1).abs(),
    'NAND_RELU': lambda x,y: 1-nn.functional.relu(x+y-1),
    'NAND_COND': lambda x,y: (x+y<=1.).to(x),
    'NAND_WHERE': nand_where,
    'OR_DEFAULT': lambda x,y: x+y-(x*y),
    'OR_COND': lambda x,y: (x+y>0.).to(x),
    'OR_WHERE': or_where,
    'NOR_DEFAULT': lambda x,y: 1-x-y+(x*y),
    'NOR_COND': lambda x,y: (x+y==0.).to(x),
    'NOR_WHERE': nor_where,
    'XOR_LONG': lambda x,y: (x+y)*(1-x*y),
    'XOR_DEFAULT': lambda x,y: x+y-(2*x*y),
    'XOR_MOD': lambda x,y: (x+y)%2,
    'XOR_COND': lambda x,y: (x+y==1.).to(x),
    'XOR_WHERE': xor_where,
    'XNOR_LONG': lambda x,y: 1-(x+y)*(1-x*y),
    'XNOR_DEFAULT': lambda x,y: 1-x-y+(2*x*y),
    'XNOR_MOD': lambda x,y: 1-((x+y)%2),
    'XNOR_COND': lambda x,y: (x+y!=1.).to(x),
    'XNOR_WHERE': xnor_where
}

def generate_activations(n=100,p=.5):
    return (torch.rand(n**2) <= p).to(torch.float32).reshape((n,n)).to(device)

def main():
    x = generate_activations(500)
    y = generate_activations(500)
    times = {}
    for cnf_name, cnf in tqdm(sew_functions.items()):
        cnf_name = tuple(cnf_name.split('_'))
        times[cnf_name] = {'time':timeit.timeit(partial(cnf,x,y),number=10000)}
    df = pd.DataFrame.from_dict(times,orient='index')
    df.index = pd.MultiIndex.from_tuples(df.index,names=['cnf','implementation'])
    df.to_csv(f'{device}_times.csv',index=True)
    print(df)
if __name__ == "__main__":
    main()
