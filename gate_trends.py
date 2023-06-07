import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm 

def sew_function(x,y,cnf='XOR'):
    if cnf == 'ADD':
        return x + y
    elif cnf == 'AND':
        return (x+y>1.).astype(float)
    elif cnf == 'IAND':
        return (x-y>0.).astype(float)
    elif cnf == 'NAND':
        return (x+y<=1.).astype(float)
    elif cnf == 'OR':
        return (x+y>0.).astype(float)
    elif cnf == 'NOR':
        return (x+y==0.).astype(float)
    elif cnf == 'XOR':
        return (x+y==1.).astype(float)
    elif cnf == 'XNOR':
        return (x+y!=1.).astype(float)
    else:
        raise NotImplementedError

def generate_activations(n=100,p=.5):
    return (np.random.rand(n**2) <= p).astype(int).reshape((n,n))

def random_generative(iters=1000,n=100,p=.5,cnf='XOR'):
    x = generate_activations(n,p=p)
    avg_active = [np.average(np.where(x>0,1,0))]
    avg_value = [np.average(x)]
    for _ in range(iters):
        y = generate_activations(n,p=np.random.rand())
        x = sew_function(x,y,cnf=cnf)
        avg_active.append(np.average(np.where(x>0,1,0)))
        avg_value.append(np.average(x))
    return np.array(avg_active), np.array(avg_value)

def independent_generative(iters=1000,n=100,p=.5,cnf='XOR'):
    x = generate_activations(n,p=p)
    avg_active = [np.average(np.where(x>0,1,0))]
    avg_value = [np.average(x)]
    for _ in range(iters):
        y = generate_activations(n,p=p)
        x = sew_function(x,y,cnf=cnf)
        avg_active.append(np.average(np.where(x>0,1,0)))
        avg_value.append(np.average(x))
    return np.array(avg_active), np.array(avg_value)

def dependent_generative(iters=1000, n=100, p=.5, cnf='XOR'):
    x = generate_activations(n,p=p)
    avg_active = [np.average(np.where(x>0,1,0))]
    avg_value = [np.average(x)]
    for _ in range(iters):
        y = generate_activations(n,p=avg_active[-1])
        x = sew_function(x,y,cnf=cnf)
        avg_active.append(np.average(np.where(x>0,1,0)))
        avg_value.append(np.average(x))
    return np.array(avg_active), np.array(avg_value)

def inverse_dependent_generative(iters=1000, n=100, p=.5, cnf='XOR'):
    x = generate_activations(n,p=p)
    avg_active = [np.average(np.where(x>0,1,0))]
    avg_value = [np.average(x)]
    for _ in range(iters):
        y = generate_activations(n,p=1-avg_active[-1])
        x = sew_function(x,y,cnf=cnf)
        avg_active.append(np.average(np.where(x>0,1,0)))
        avg_value.append(np.average(x))
    return np.array(avg_active), np.array(avg_value)

def viz_test(test, ps, avg_actives, avg_values, cnf='XOR'):
    iters = len(avg_actives[0])

    fig, axs = plt.subplots(2,figsize=(20,15))
    fig.suptitle(f'{cnf} Over Time For {test.__name__}')
    axs[0].set_title('Activation Percentage Over Time')
    axs[0].set_ylim((0,100))
    axs[1].set_title('Average Value Over Time')
    axs[1].set_ylim((0,(iters if cnf=='ADD' else 1)))

    X = np.arange(iters)

    for idx in range(len(ps)):
        axs[0].plot(X, avg_actives[idx]*100, label=f'P(X)={ps[idx]*100:.2f}%')
        axs[1].plot(X, avg_values[idx], label=f'P(X)={ps[idx]*100:.2f}%')

    axs[0].legend(title='Initial Activation Probability')
    axs[0].set(xlabel='Iterations',ylabel='Activation Percentage')
    axs[1].legend(title='Initial Activation Probability')
    axs[1].set(xlabel='Iterations',ylabel='Average Value')

    return fig, axs

def run_test(test,iters=500,n=100,ps=None,cnf='XOR'):
    if ps is None:
        ps = np.linspace(0,1,10,endpoint=False)[1:]

    avg_actives=[]
    avg_values=[]

    for p in tqdm(ps):
        avg_active, avg_value = test(iters=iters,n=n,p=p,cnf=cnf)
        avg_actives.append(avg_active)
        avg_values.append(avg_value)
    
    fig, axs = viz_test(test,ps,avg_actives,avg_values,cnf=cnf)
    path = os.path.join('figures',cnf,f'{test.__name__}.png')
    os.makedirs(os.path.dirname(path),exist_ok=True)
    plt.savefig(path)
    plt.close()
    plt.clf()

TESTS = [independent_generative,dependent_generative,random_generative,inverse_dependent_generative]
CNFS = ['ADD','AND','IAND','NAND','OR','NOR','XOR','XNOR']
PS = [0.01,0.02,0.05,0.1,0.3,0.5,0.7,0.9]

def main():
    for test in TESTS:
        for cnf in CNFS:
            run_test(test,n=200,iters=100,cnf=cnf,ps=PS)

if __name__ == "__main__":
    main()