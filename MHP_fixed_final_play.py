import numpy as np
import time as T
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.extmath import cartesian
import matplotlib.pyplot as plt

import sys


def generate_seq_func(mu,omega,alpha,horizon, seed = None):
    if seed is not None:
        np.random.seed(seed)
    
    data = []
    Istar = np.sum(mu)
    s = np.random.exponential(scale=1./Istar)

    # attribute (weighted random sample, since sum(mu)==Istar)
    dim = mu.shape[0]
    n0 = np.random.choice(np.arange(dim), p=(mu / Istar)).item()  #这个相当于根据mu的归一化大小，来选择初始事件是哪个
    data.append([s, int(n0)])  #第一次事件是必然发生，所以不需要采样拒绝
    
    # value of λ(t_k) where k is most recent event
    lastrates = mu.copy()
    decIstar = False   #检查上一次是否有declim, 由于之前提到
    
    while True:
        tj, uj = data[-1][0], int(data[-1][1])  #拿到最新的时间戳和事件类型

        if decIstar:
            Istar = np.sum(rates)    #如果上次是拒绝的，那么就沿用上次，相当于步进没有成功，
            decIstar = False
        else:                        #如果上次成功了，就要更新下次步进的上限，
            Istar = np.sum(lastrates) + omega * np.sum(alpha[:,uj])   
            
        s += np.random.exponential(scale=1./Istar)  #候选事件的时间”，根据上界强度 I*进行模拟，只是针对上限而已，不一定会发生
        
        rates = mu + np.exp(-omega * (s - tj)) * \
                    (alpha[:,uj].flatten() * omega + lastrates - mu)   #通过递归的方式来描述强度
        
        
        diff = Istar - np.sum(rates)   #diff 相当于来个空缺概率
        try:   #这里的意思是选一个事件，相当于n0代表是哪个事件。
            n0 = np.random.choice(np.arange(dim+1), p=(np.append(rates, diff) / Istar)).item()  #n0得到触发事件，append出来
        except ValueError:
            print('Probabilities do not sum to one.')
            data = np.array(data)
            return data
        
        if n0 < dim:
            data.append([s, int(n0)])
            lastrates = rates.copy()
        else:
            decIstar = True

        if s >= horizon:
            data = np.array(data)
            data = data[data[:,0] < horizon]
            return data
    
def EM_func(Ahat, mhat, omega, seq=[], smx=None, tmx=None, regularize=False, 
           Tm=-1, maxiter=100, epsilon=0.01, verbose=True):
    
    if len(seq) == 0:
        return

    N = len(seq)
    dim = mhat.shape[0]
    Tm = float(seq[-1,0]) if Tm < 0 else float(Tm)
    sequ = seq[:,1].astype(int)

    p_ii = np.random.uniform(0.01, 0.99, size=N)   #这两步相当于在初始化责任概率,注意这里维度是N，也就是数据的所有长度
    p_ij = np.random.uniform(0.01, 0.99, size=(N, N))

    diffs = pairwise_distances(np.array([seq[:,0]]).T, metric = 'euclidean') #计算事件时间差
    diffs[np.triu_indices(N)] = 0  #只取矩阵的下半角
    kern = omega * np.exp(-omega * diffs) #kernal核函数，其实从这里可以看出来，omega作为超参数的话，似乎能方便优化

    colidx = np.tile(sequ.reshape((1,N)), (N,1))  # 每列是 u_j，反正每一行都是一样的
    rowidx = np.tile(sequ.reshape((N,1)), (1,N))    # 每行是 u_i，反正每一列都是一样的
    seqcnts = np.array([len(np.where(sequ==i)[0]) for i in range(dim)])  #统计每一个事件触发了多少次
    seqcnts = np.tile(seqcnts, (dim,1))  # 把上述信息复制成shape=(dim, dim) 的二维矩阵，每一行都相同。
    
    def sum_pij(a,b):
        
        # 输入a和b代表的是事件的类型，这里相当于把所有1触发0的历史时间对，都给弄出来
        c = cartesian([np.where(seq[:,1]==int(a))[0], np.where(seq[:,1]==int(b))[0]])  #这里把a和b事件的时间所有两两全排列都弄出来
        return np.sum(p_ij[c[:,0], c[:,1]])  #把所有所选事件
    vp = np.vectorize(sum_pij)  #这个向量化版本方便以后操作
    
    k = 0 #迭代次数
    old_LL = -10000  # 上一次的log likelihood，用于判断是否收敛
    
    while k < maxiter:
        
        ## --- E 步：更新 p_ij, p_ii ---
        Auu = Ahat[rowidx, colidx]  
        ag = np.multiply(Auu, kern)  #Auu在形状上要匹配上kern，也就是说针对每个历史数据中的时间对都要有描述
        ag[np.triu_indices(N)] = 0

        mu = mhat[sequ]   # 这个只是一维的
        rates = mu + np.sum(ag, axis=1)  # 广播相加

        p_ij = np.divide(ag, np.tile(np.array([rates]).T, (1,N))) #p_ij = ag / rates[:, None]   # shape: (N,N)
        p_ii = np.divide(mu, rates)  #p_ii = mu / rates            # shape: (N,)
        
        
        # --- M 步：更新 mhat 和 Ahat ---
        mhat = np.array([np.sum(p_ii[np.where(seq[:,1]==i)]) for i in range(dim)]) / Tm  #这段代码通过责任化概率形式后的LL求导，给出了估计m的公式

        if regularize:
            Ahat = np.divide(np.fromfunction(lambda i,j: vp(i,j), (dim,dim)) + (smx-1), seqcnts + tmx)
        else:
            Ahat = np.divide(np.fromfunction(lambda i,j: vp(i,j), (dim,dim)), seqcnts)  #与m推导类似，但似乎更复杂了一些
        
        
        # --- LL 收敛判断（每10次打印） ---
        if k % 10 == 0:
            try:
                term1 = np.sum(np.log(rates))
            except:
                print('Log error!')
            term2 = Tm * np.sum(mhat)
            term3 = np.sum(np.sum(Ahat[u,int(seq[j,1])] for j in range(N)) for u in range(dim))
            new_LL = (1./N) * (term1 - term3)
            if abs(new_LL - old_LL) <= epsilon:
                if verbose:
                    print('Reached stopping criterion. (Old: %1.3f New: %1.3f)' % (old_LL, new_LL))
                return Ahat, mhat
            if verbose:
                print('After ITER %d (old: %1.3f new: %1.3f)' % (k, old_LL, new_LL))
                print(' terms %1.4f, %1.4f, %1.4f' % (term1, term2, term3))
            old_LL = new_LL
        k += 1

    if verbose:
        print('Reached max iter (%d).' % maxiter)

    Ahat = Ahat
    mhat = mhat
    return Ahat, mhat
    


    
    
class MHP:
    def __init__(self, alpha=[[0.5]], mu=[0.1], omega=1.0):
        '''params should be of form:
        alpha: numpy.array((u,u)), mu: numpy.array((,u)), omega: float'''
        self.data = []
        self.alpha, self.mu, self.omega = np.array(alpha), np.array(mu), omega
        self.dim = self.mu.shape[0]
        self.check_stability()
    def check_stability(self):
        ''' check stability of process (max alpha eigenvalue < 1)'''
        w,v = np.linalg.eig(self.alpha)
        me = np.amax(np.abs(w))
        print('Max eigenvalue: %1.5f' % me)
        if me >= 1.:
            print('(WARNING) Unstable.')

    def generate_seq(self, horizon, seed=None):
        '''Generate a sequence based on mu, alpha, omega values. 
        Uses Ogata's thinning method, with optional random seed for reproducibility'''
        if seed is not None:
            np.random.seed(seed)

        self.data = []  # clear history

        Istar = np.sum(self.mu)
        s = np.random.exponential(scale=1./Istar)

        # attribute (weighted random sample, since sum(mu)==Istar)
        n0 = np.random.choice(np.arange(self.dim), p=(self.mu / Istar)).item()
        self.data.append([s, int(n0)])

        # value of λ(t_k) where k is most recent event
        lastrates = self.mu.copy()
        decIstar = False

        while True:
            tj, uj = self.data[-1][0], int(self.data[-1][1])

            if decIstar:
                Istar = np.sum(rates)
                decIstar = False
            else:
                Istar = np.sum(lastrates) + self.omega * np.sum(self.alpha[:,uj])

            s += np.random.exponential(scale=1./Istar)

            rates = self.mu + np.exp(-self.omega * (s - tj)) * \
                    (self.alpha[:,uj].flatten() * self.omega + lastrates - self.mu)

            diff = Istar - np.sum(rates)
            try:
                n0 = np.random.choice(np.arange(self.dim+1), p=(np.append(rates, diff) / Istar)).item()
            except ValueError:
                print('Probabilities do not sum to one.')
                self.data = np.array(self.data)
                return self.data

            if n0 < self.dim:
                self.data.append([s, int(n0)])
                lastrates = rates.copy()
            else:
                decIstar = True

            if s >= horizon:
                self.data = np.array(self.data)
                self.data = self.data[self.data[:,0] < horizon]
                return self.data

    def EM(self, Ahat, mhat, omega, seq=[], smx=None, tmx=None, regularize=False, 
           Tm=-1, maxiter=100, epsilon=0.01, verbose=True):
        if len(seq) == 0:
            seq = self.data

        N = len(seq)
        dim = mhat.shape[0]
        Tm = float(seq[-1,0]) if Tm < 0 else float(Tm)
        sequ = seq[:,1].astype(int)

        p_ii = np.random.uniform(0.01, 0.99, size=N)
        p_ij = np.random.uniform(0.01, 0.99, size=(N, N))

        diffs = pairwise_distances(np.array([seq[:,0]]).T, metric = 'euclidean')
        diffs[np.triu_indices(N)] = 0
        kern = omega * np.exp(-omega * diffs)

        colidx = np.tile(sequ.reshape((1,N)), (N,1))
        rowidx = np.tile(sequ.reshape((N,1)), (1,N))
        seqcnts = np.array([len(np.where(sequ==i)[0]) for i in range(dim)])
        seqcnts = np.tile(seqcnts, (dim,1))

        def sum_pij(a,b):
            c = cartesian([np.where(seq[:,1]==int(a))[0], np.where(seq[:,1]==int(b))[0]])
            return np.sum(p_ij[c[:,0], c[:,1]])
        vp = np.vectorize(sum_pij)

        k = 0
        old_LL = -10000
        while k < maxiter:
            Auu = Ahat[rowidx, colidx]
            ag = np.multiply(Auu, kern)
            ag[np.triu_indices(N)] = 0

            mu = mhat[sequ]
            rates = mu + np.sum(ag, axis=1)

            p_ij = np.divide(ag, np.tile(np.array([rates]).T, (1,N)))
            p_ii = np.divide(mu, rates)

            mhat = np.array([np.sum(p_ii[np.where(seq[:,1]==i)]) for i in range(dim)]) / Tm

            if regularize:
                Ahat = np.divide(np.fromfunction(lambda i,j: vp(i,j), (dim,dim)) + (smx-1), seqcnts + tmx)
            else:
                Ahat = np.divide(np.fromfunction(lambda i,j: vp(i,j), (dim,dim)), seqcnts)

            if k % 10 == 0:
                try:
                    term1 = np.sum(np.log(rates))
                except:
                    print('Log error!')
                term2 = Tm * np.sum(mhat)
                term3 = np.sum(np.sum(Ahat[u,int(seq[j,1])] for j in range(N)) for u in range(dim))
                new_LL = (1./N) * (term1 - term3)
                if abs(new_LL - old_LL) <= epsilon:
                    if verbose:
                        print('Reached stopping criterion. (Old: %1.3f New: %1.3f)' % (old_LL, new_LL))
                    return Ahat, mhat
                if verbose:
                    print('After ITER %d (old: %1.3f new: %1.3f)' % (k, old_LL, new_LL))
                    print(' terms %1.4f, %1.4f, %1.4f' % (term1, term2, term3))
                old_LL = new_LL
            k += 1

        if verbose:
            print('Reached max iter (%d).' % maxiter)

        self.Ahat = Ahat
        self.mhat = mhat
        return Ahat, mhat

    def get_rate(self, ct, d):
        seq = np.array(self.data)
        if not np.all(ct > seq[:,0]): seq = seq[seq[:,0] < ct]
        return self.mu[d] + \
            np.sum([self.alpha[d,int(j)]*self.omega*np.exp(-self.omega*(ct-t)) for t,j in seq])

    def plot_rates(self, horizon=-1):
        if horizon < 0:
            horizon = np.amax(self.data[:,0])

        f, axarr = plt.subplots(self.dim*2,1, sharex='col', 
                                gridspec_kw = {'height_ratios':sum([[3,1] for i in range(self.dim)],[])}, 
                                figsize=(8,self.dim*2))
        xs = np.linspace(0, horizon, 1000)
        for i in range(self.dim):
            row = i * 2
            r = [self.get_rate(ct, i) for ct in xs]
            axarr[row].plot(xs, r, 'k-')
            axarr[row].set_ylim([-0.01, np.amax(r)+(np.amax(r)/2.)])
            axarr[row].set_ylabel('$\lambda(t)_{%d}$' % i, fontsize=14)
            r = []
            subseq = self.data[self.data[:,1]==i][:,0]
            axarr[row+1].plot(subseq, np.zeros(len(subseq)) - 0.5, 'bo', alpha=0.2)
            axarr[row+1].yaxis.set_visible(False)
            axarr[row+1].set_xlim([0, horizon])
        plt.tight_layout()

    def plot_events(self, horizon=-1, showDays=True, labeled=True):
        if horizon < 0:
            horizon = np.amax(self.data[:,0])
        fig = plt.figure(figsize=(10,2))
        ax = plt.gca()
        for i in range(self.dim):
            subseq = self.data[self.data[:,1]==i][:,0]
            plt.plot(subseq, np.zeros(len(subseq)) - i, 'bo', alpha=0.2)
        if showDays:
            for j in range(1,int(horizon)):
                plt.plot([j,j], [-self.dim, 1], 'k:', alpha=0.15)
        if labeled:
            ax.set_yticklabels('')
            ax.set_yticks(-np.arange(0, self.dim), minor=True)
            ax.set_yticklabels([r'$e_{%d}$' % i for i in range(self.dim)], minor=True)
        else:
            ax.yaxis.set_visible(False)
        ax.set_xlim([0,horizon])
        ax.set_ylim([-self.dim, 1])
        ax.set_xlabel('Days')
        plt.tight_layout()


sys.exit()


'''
which will initialize a univariate process with parameters mu=[0.1], alpha=[[0.5]], and omega=1.0. 
This sequence is stored as P.data, a numpy.ndarray with 2 columns: the first column with the timestamps, 
the second with the stream assignment (in this case there is only one stream).
'''
# P = MHP()
# data = P.generate_seq(1000, seed=42)   
# # P.plot_events()
# print(data[:])

'''

dim=3
We can also look at the conditional intensities along with the time series of events:
'''

m = np.array([0.2, 0.0, 0.0])
a = np.array([[0.1, 0.0, 0.0], 
              [0.9, 0.0, 0.0],
              [0.0, 0.9, 0.0]])
w = 3.1

P = MHP(mu=m, alpha=a, omega=w)
data3=P.generate_seq(6000,seed=42)

data3_func = generate_seq_func(mu=m,omega=w,alpha=a,horizon=6000, seed = 42)

# P.plot_events()





'''
还可以进行参数估计
'''
np.random.seed(42)

mhat = np.random.uniform(0,1, size=3)
ahat = np.random.uniform(0,1, size=(3,3))
w = 3.

print('initial parameters:')
print('m:',m)
print('ahat:',a)


# temp=P.EM(ahat, mhat, w)  #这个会让

temp = P.EM(ahat, mhat, w, seq=P.data[:])
temp_data3 = P.EM(ahat, mhat, w, seq=data3_func)
temp_para = EM_func(ahat,mhat,w,seq=data3_func)

print(temp)
print(temp_data3)
print(temp_para)


