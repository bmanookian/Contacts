import numpy as np
from functools import partial
import itertools
import multiprocessing
import csv
import sys
import pickle
sys.path.append('/home/bmanookian/Contacts')
import shortenRes as sR

def datareader(inputfile):
    out = []
    with open(inputfile, newline = '') as file:
        reader = csv.reader(file)
        for i,row in enumerate(reader):
            out.append(row)
    return np.array(out)

def datawrite(output,data,labels=None):
    with open(output, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if labels is not None:
            csv_writer.writerow(labels)
        for row in data:
            csv_writer.writerow(row)

def picklewrite(output,data):
    with open(output, 'wb') as file:
        pickle.dump(data, file)

def pickleread(picklefile):
    with open(picklefile, 'rb') as file:
        return pickle.load(file)

def runParallel(foo,iter,ncore):
    pool=multiprocessing.Pool(processes=ncore)
    try:
        out=(pool.map_async( foo,iter )).get()  
    except KeyboardInterrupt:
        print ("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
        pool.join()
    else:
        #print ("Quitting normally core used ",ncore)
        pool.close()
        pool.join()
    try:
        return out
    except Exception:
        return out



def encode(c):
    try:
        b=np.ones(c.shape[1],dtype=int)
    except Exception:
        c=np.column_stack(c)
        b=np.ones(c.shape[1],dtype=int)
    b[:-1]=np.cumprod((c[:,1:].max(0)+1)[::-1])[::-1]
    return np.sum(b*c,1)

def mi_p(A):
    X,Y=A
    return H(X)+H(Y)-joinH((X,Y))
def H(i):
        """entropy of labels"""
        p=np.unique(i,return_counts=True)[1]/i.size
        return -np.sum(p*np.log2(p))

def joinH(i):
	pair=np.column_stack((i))
	en=encode(pair)
	p=np.unique(en,return_counts=True)[1]/len(en)
	return -np.sum(p*np.log2(p))


def read_tsv(fin):
    f=open(fin)
    f.readline()
    h=f.readline().strip().split('\t')
    data=np.array([r.strip().split('\t') for r in f.readlines()])
    return h, data


def get_unique_pair2(data):
    pair=np.array([':'.join(a.split(':')[1:-1]+b.split(':')[1:-1]) for a,b in data[:,2:]])
    u,pair_indx=np.unique(pair,return_inverse=True)
    return pair,u,pair_indx

def transformRes(pair):
    R=sR.Residue()
    res1=R.shRes(pair.split('_')[0][:3])+pair.split('_')[0][3:]
    res2=R.shRes(pair.split('_')[1][:3])+pair.split('_')[1][3:]
    return res1+'_'+res2

def get_unique_pair(data):
    P=np.array([a.split(':')[1:3][0]+a.split(':')[1:3][1][:]+'_'+b.split(':')[1:3][0]+b.split(':')[1:3][1][:] for a,b in data[:,2:]])
    Pt=np.array([transformRes(p) for p in P])
    u,pair_indx=np.unique(Pt,return_inverse=True)
    return Pt,u,pair_indx

def remove_Neighbors(contacts,N):
    iplus=np.array([i for i,p in enumerate(contacts) if int(p.split('_')[0][1:])+N != int(p.split('_')[1][1:])])
    imin=np.array([i for i,p in enumerate(contacts) if int(p.split('_')[0][1:])-N != int(p.split('_')[1][1:])])
    return np.intersect1d(iplus,imin)

def get_traj(u,T,pair):
    tmax=T[-1]
    traj=np.zeros((tmax+1,u.size))
    for ic,ui in enumerate(u):
        traj[ic,np.unique(T[pair==ui])]=1
    return  traj   
def get_trj_s(pair_indx,T,ui):
    tmax=T[-1]
    traj=np.zeros(tmax+1)
    traj[np.unique(T[pair_indx==ui])]=1
    return traj


def get_traj_p(T,pair_indx): #u also
    tmax=T[-1]
    u=np.unique(pair_indx)
    traj=np.zeros((tmax+1,u.size))
    make_v=partial(get_trj_s,pair_indx,T)
    trj=runParallel(make_v,u,30)
    return trj




class traj_from_contact():
    
    def __init__(self,fin=None,direct='',traj=None,unqpair=None):
        if fin is not None:
            h,data=read_tsv(fin)
            np.save(f'{direct}/h',h)
            np.save(f'{direct}/data',data)
            self.pair,self.input_unqpair,self.pair_indx=get_unique_pair(data)
            T=data[:,0].astype(int)
            self.input_traj=np.array(get_traj_p(T,self.pair_indx))
        if traj is not None:
            self.input_traj=traj
            self.input_unqpair=unqpair
        self.traj=self.input_traj
        self.unqpair=self.input_unqpair    
    
    def restore_input_traj(self):
        self.traj=self.input_traj
        self.unqpair=self.input_unqpair

    def cuttraj(self,start,end):
        self.traj=self.traj[:,start:end]

    def remove_singles(self):
        self.iv=[i for i,x in enumerate(self.traj) if len(set(x))>1]
        self.traj=self.traj[self.iv]
        self.unqpair=self.unqpair[self.iv]

    def remove_Neighbor(self,N):
        I=remove_Neighbors(self.unqpair,N)
        self.unqpair=self.unqpair[I]
        self.traj=self.traj[I]
        
        
    
    def compute_MI_matrix(self,numproc,MI=None):
        if MI is None:
            self.MI=runParallel(mi_p,itertools.combinations(self.traj,2),numproc)
        if MI is not None:
            self.MI=MI
        D=np.zeros((self.traj.shape[0],self.traj.shape[0]))
        D[np.triu_indices_from(D,1)]=self.MI
        self.D=D+D.T
    
    def find_pairs_to_remove(self,th):
        self.indx_pair2remove=np.all(self.D<th,1)
        self.pair2remove=self.unqpair[self.indx_pair2remove]
        self.unqpair=self.unqpair[~self.indx_pair2remove]
        self.traj=self.traj[~self.indx_pair2remove]
        self.DValid=self.D[~self.indx_pair2remove][:,~self.indx_pair2remove]

