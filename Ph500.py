from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import random

data_all = loadmat('Ph500.mat')
M1 = (data_all['Ph500']).todense()
n = len(M1)
M = [[M1[i,j] for j in range(n)] for i in range(n)]

plt.figure(figsize=(12,10))
plt.imshow(M, cmap='plasma', interpolation='nearest')
plt.colorbar(label='Matrix value')
plt.title('Matrix (colored by entries)')
plt.xlabel('Column index')
plt.ylabel('Row index')
plt.show()


def ima_smisla(n1, n2, n3, M):
    global min_size
    if n2-n1 < min_size or n3-n2 < min_size:
        return False
        
    sumadiag = sum([sum(M[i][n1:n2+1]) for i in range(n1, n2+1)])/(n2-n1+1) + sum([sum(M[i][n2:n3+1]) for i in range(n2, n3+1)])/(n3-n2+1)
    sumanediag = sum([sum(M[i][n1:n2+1]) for i in range(n2, n3+1)])/(n2-n1+1) + sum([sum(M[i][n2:n3+1]) for i in range(n1, n2+1)])/(n3-n2+1)
    statistika = sumanediag / sumadiag
    global delta
    return statistika < delta

def podijeli(poc, kraj):
    global n, min_size
    
    if kraj-poc<min_size or poc<0 or kraj>=n:
        return
    
    global M
    n1=kraj-poc+1
    
    M1 = [[M[poc+i][poc+j] for j in range(n1)] for i in range(n1)]
    U, S, Vt = np.linalg.svd(M1)
    
    cutoff = 0

    perm = list()
    if poc != 0:
        perm = perm + list(range(poc))
    for j in range(poc, kraj+1):
        if np.sign(U[j-poc][1])==1:
            perm.append(j)
    for j in range(poc, kraj+1):
        if np.sign(U[j-poc][1])==0:
            perm.append(j)
    cutoff = len(perm)
    for j in range(poc, kraj+1):
        if np.sign(U[j-poc][1])==-1:
            perm.append(j)
    if kraj != n-1:
        perm = perm + list(range(kraj+1, n))
            
    nas_fix_M = [[M[perm[i]][perm[j]] for j in range(n)] for i in range(n)]
    
    if ima_smisla(poc, cutoff, kraj, nas_fix_M):
        M = nas_fix_M.copy()
        podijeli(poc, cutoff)
        podijeli(cutoff+1, kraj)
    else:
        global broj_klastera
        global gid
        broj_klastera += 1
        for i in range(poc, kraj+1):
            gid[i] = broj_klastera
        print(poc, kraj)

        

def moj_k_means():
    global broj_klastera, centri, gid
    
    for i in range(1, broj_klastera+1):
        for j in range(n):
            centri[i][j] = sum([(M[k][j]) if gid[k]==i else 0 for k in range(n)]) / gid.count(i)

    for loop_number in range(100):
        promjena=0
        for i in range(n):
            idx, val = 1, np.linalg.norm([centri[1][j] - M[i][j] for j in range(n)])
            for k in range(2, broj_klastera+1):
                tmp = np.linalg.norm([centri[k][j] - M[i][j] for j in range(n)])
                if tmp < val:
                    val = tmp
                    idx = k
            if (gid[i] != idx):
                gid[i] = idx
                promjena = 1

        if not promjena:
            break
        
        for i in range(1, broj_klastera+1):
            for j in range(n):
                if gid.count(i)!=0:
                    centri[i][j] = sum([(M[k][j]) if gid[k]==i else 0 for k in range(n)]) / gid.count(i)

    return

def traveling_salesman():
    global salesman
    posjeceni = [0, 1]+([0]*(broj_klastera-1)) # posjetio sam samo 1

    for i in range(broj_klastera-1):
        trenutni = salesman[-1]
        iduci = -1

        for j in range(2, broj_klastera+1):
            if not posjeceni[j]:
                iduci = j
                break
        min_dist = np.linalg.norm([centri[trenutni][k] - centri[iduci][k] for k in range(n)])

        for j in range(iduci+1, broj_klastera+1):
            if not posjeceni[j]:
                temp = np.linalg.norm([centri[trenutni][k] - centri[j][k] for k in range(n)])
                if temp < min_dist:
                    min_dist = temp
                    iduci = j

        posjeceni[iduci] = 1
        salesman.append(iduci)

    return
def evaluacija(M, gid):
    n = len(M)
    gid = np.array(gid)
    K = np.max(gid)

    # --- Metastability Index ---
    pi = np.sum(M, axis=0) / np.sum(M)
    msi = 0.0
    for k in range(1, K + 1):
        idx = np.where(gid == k)[0]
        if len(idx) == 0:
            continue
        pi_k = np.sum(pi[idx])
        if pi_k == 0:
            continue
        P_kk = np.sum([pi[i] * np.sum([M[i][j] for j in idx]) for i in idx]) / pi_k
        msi += pi_k * P_kk

    # --- Inter/Intra Density Ratio ---
    ratios = []
    for k in range(1, K + 1):
        idx_in = np.where(gid == k)[0]
        idx_out = np.where(gid != k)[0]
        if len(idx_in) == 0 or len(idx_out) == 0:
            continue
        intra = sum([M[i][j] for i in idx_in for j in idx_in]) / (len(idx_in) ** 2)
        inter = (sum([M[i][j] for i in idx_in for j in idx_out]) + sum([M[i][j] for i in idx_out for j in idx_in])) / (
                    2 * len(idx_in) * len(idx_out))
        if intra > 0:
            ratios.append(inter / intra)
    inter_intra_ratio = sum(ratios) / len(ratios) if ratios else float('inf')

    print(f"  MSI: {msi:.4f}")
    print(f"  Inter/Intra Ratio: {inter_intra_ratio:.4f}")
    

# MAIN

for step in range(10):
    broj_klastera = 0
    gid = [0] * n
    min_size = 13 - step  # najmanja dopuštena veličina klastera
    delta = step + 0.5  # tolerancija za funkciju ima_smisla
    print("Korak:", step)
    # SVD da vidim razumni broj klastera i inicijalnu podijelu
    podijeli(0, n - 1)
    print("svd_samo")
    evaluacija(M, gid)
    # K means da poboljšam očite pripadnosti drugim klasterima
    centri = [[0] * n for _ in range(broj_klastera + 1)]
    moj_k_means()
    print("K-means + svd")
    evaluacija(M, gid)
    # I sad još grupiram one koji su "najbliže"; samo greedy algoritam
    salesman = [1]
    traveling_salesman()

    perm = []
    for i in range(broj_klastera):
        for j in range(n):
            if gid[j] == salesman[i]:
                perm.append(j)

    novi_M = [[M[perm[i]][perm[j]] for j in range(n)] for i in range(n)]
    M = novi_M.copy()

    plt.figure(figsize=(12, 10))
    plt.imshow(M, cmap='plasma', interpolation='nearest')
    plt.colorbar(label='Matrix value')
    plt.title('Matrix (colored by entries)')
    plt.xlabel('Column index')
    plt.ylabel('Row index')
    plt.show()

    print("Korak:", step, "\nBroj klastera", broj_klastera)
