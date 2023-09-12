import numpy as np
import random

#@title : Adaptive Block Bregman Kaczmarz methods

def kaczmarz_method(A, x_start, b, x_true, max_iter, sigma_list, gamma, lbda, nber_block, p_list=[], beta=0, save_freq = 1, lr= 'css', keep = False, shuffle=False):

  # the adaptive sparse Kaczmarz method
  # for academic purpose; uses x_true to calculate the exact beta needed for the adaptive stepsize
  # A: The given matrix.
  # x_start: The starting point of the algorithm.
  # b: The true right hand size. We never used it and it is only to get a noisy version of itself.
  # max_iter: The maximum number of iterations.
  # sigma_list: The list of noise variance of every block.
  # gamma: The convergence rate or an estimate thereof - needed to compute the adaptive stepsize
  # lbda: The sparsity parameter.
  # nber_block: The number of blocks the user want to use.
  # p_list: The partition of the matrix A.
  # beta: Parameter of the stepsize or an estimate thereof - needed to compute the adaptive stepsize
  # save_freq: The frequence at which we want to save the residual and the errors.
  # lr: learning rate/stepsize.
  #     Options: css - constant stepsize, i.e. standard sparse Kaczmarz. Does not need gamma and beta
  #              oss - optimal stepsize according to paper. Needs gamma and beta!
  #              tbdss - time based decaying stepsize. stepsize decays like 1/k. Only needs gamma
  # keep: "True" if we want to save the dual iterates after save_freq iteration.
  # shuffle: If set to "True" take the partition in a random way.

  m, n = A.shape
  if x_start.shape != (n,1):
    x_start = x_start.reshape(n,1)
  if x_true.shape != (n,1):
    x_true = x_true.reshape(n,1)
  if b.shape != (m,1):
    b = b.reshape(m,1)
  if len(sigma_list) != nber_block:
     raise Exception(f'Mismatch between the length of sigma {len(sigma_list)} and the number of block {nber_block}')

  # Initialize variables
  x_star = np.zeros((n, 1))
  x_k = np.zeros((n, 1))

  residuals = []
  errors = []
  lr_values = []
  beta_values = []

  if keep == True :   # If we want to save the dual iterates
    dual_iterates = np.zeros((max_iter+1, n, 1))
    dual_iterates[0] = x_star

  # Fixing the partitions for all iterations and the probabilities vectors
  if len(p_list) == 0:
    index_list = list([i for i in range(m)])
    copy_index_list = index_list.copy()
    if shuffle == True:
      np.random.shuffle(copy_index_list)   # comment this line if you want to partition the matrix in a consecutive rows.
    p_list = np.array_split(copy_index_list, nber_block)
  squared_block_row_norms = []
  for blc in range(nber_block):
    idx = p_list[blc]
    idx.sort()
    squared_block_row_norms.append(np.linalg.norm(A[idx], ord=2)**2)
  probabilities = [norm/sum(squared_block_row_norms) for norm in squared_block_row_norms]

  residuals.append(np.linalg.norm(A @ x_k - b)/ (np.linalg.norm(b)))
  errors.append(np.linalg.norm(x_k - x_true)/ (np.linalg.norm(x_true)))

  if beta ==0: # we compute the true beta_0 using the true solution if beta is not given.
    if lbda == 0:
      print(f'Computing beta_0 for lambda={lbda}')
      beta = (sum(squared_block_row_norms) * (np.linalg.norm(x_start - x_true)**2))/ sum([sigma**2 for sigma in sigma_list])
    elif lbda > 0:
      print(f'Computing beta_0 for lambda={lbda}')
      beta = (sum(squared_block_row_norms) * (Bregman_distance_dual(x_start, x_true, lbda)))/ sum([sigma**2 for sigma in sigma_list])

  for iter in range(max_iter):
    
    beta_values.append(beta)

    # sample the block row index
    tauk = random.choices([j for j in range(nber_block)], weights = probabilities, k = 1)
    i = tauk[0]
    index = p_list[i]
    index.sort()

    if lr == 'oss' :
      if iter == 0 :
        print(f'Algorithm 1 : stepsize {lr} = Optimal stepsize, M = {nber_block}, shuffle = {shuffle}, gamma = {gamma}, lambda = {lbda} for {max_iter} iterations')

      eta = gamma * beta /(gamma * beta + 1)   

    elif lr == 'tbdss':
      if iter == 0 :
        print(f'Algorithm 1 : stepsize {lr} = Time-based decay stepsize, M = {nber_block}, shuffle = {shuffle}, gamma = {gamma}, lambda = {lbda} for {max_iter} iterations')
      eta = 2/(2 + gamma*iter)  

    elif lr == 'css' :
      if iter == 0 :
        print(f'Algorithm 1 : stepsize {lr} = Constant stepsize, M = {nber_block}, shuffle = {shuffle}, gamma = {gamma}, lambda = {lbda} for {max_iter} iterations')
      eta = 1

    # Bregman-Kaczmarz update: Algorithm 1
    # get a fresh noisy sample from the rhs
    b_noisy = b[index] + np.random.normal(0,sigma_list[i]/np.sqrt(index.size),index.size).reshape(b[index].shape)
    # calculate update
    upd = (A[index].T @ ( A[index] @ x_k - b_noisy)) / squared_block_row_norms[i]
    x_star -= eta * upd
    x_k = soft_skrinkage(x_star, lbda)

    if lbda > 0 :
      beta_new = beta*(1 - 0.5* gamma * eta)   
    elif lbda == 0 :
      beta_new = beta*(1 - gamma * eta)
    beta = beta_new

    if (iter+1) % save_freq == 0 :
      if keep == True :
        dual_iterates[iter+1] = x_star
      lr_values.append(eta)

      residuals.append(np.linalg.norm(A @ x_k - b)/ (np.linalg.norm(b)))
      errors.append(np.linalg.norm(x_k - x_true)/ (np.linalg.norm(x_true)))
  if keep == True :
    return x_k, residuals, errors, lr_values, beta_values, dual_iterates
  else :
    return x_k, residuals, errors, lr_values, beta_values

def myphantom(N):
  # Adapted from:
  # Peter Toft, "The Radon Transform - Theory and Implementation", PhD
  # thesis, DTU Informatics, Technical University of Denmark, June 1996.
  # Translated from MATLAB to Python by ChatGPT
  xn = ((np.arange(N) - (N - 1) / 2) / ((N - 1) / 2)).reshape(-1, 1)
  Xn = np.tile(xn, (1, N))
  Yn = np.rot90(Xn)
  X = np.zeros((N, N))

  e = np.array([
    [1, 0.69, 0.92, 0, 0, 0],
    [-0.8, 0.6624, 0.8740, 0, -0.0184, 0],
    [-0.2, 0.1100, 0.3100, 0.22, 0, -18],
    [-0.2, 0.1600, 0.4100, -0.22, 0, 18],
    [0.1, 0.2100, 0.2500, 0, 0.35, 0],
    [0.1, 0.0460, 0.0460, 0, 0.1, 0],
    [0.1, 0.0460, 0.0460, 0, -0.1, 0],
    [0.1, 0.0460, 0.0230, -0.08, -0.605, 0],
    [0.1, 0.0230, 0.0230, 0, -0.606, 0],
    [0.1, 0.0230, 0.0460, 0.06, -0.605, 0]
  ])

  for i in range(e.shape[0]):
    a2 = e[i, 1] ** 2
    b2 = e[i, 2] ** 2
    x0 = e[i, 3]
    y0 = e[i, 4]
    phi = np.radians(e[i, 5])
    A = e[i, 0]
    x = Xn - x0
    y = Yn - y0
    index = np.where(
      ((x * np.cos(phi) + y * np.sin(phi)) ** 2) / a2
      + ((y * np.cos(phi) - x * np.sin(phi)) ** 2) / b2
      <= 1
    )
    X[index] += A

  X = X.ravel()
  X[X < 0] = 0
  return X

# The soft shrinkage function
def soft_skrinkage(x, lbda):
  return np.sign(x) * np.maximum(np.abs(x) - lbda, 0)

# The Bregman distance function
def Bregman_distance_dual(z, y, lbda):
  p1 = dual_objective(z, lbda)
  p2 = np.dot(z.T, y)
  p3 = objective(y, lbda)
  return (p1-p2+p3)[0][0]

# The regularized l_1 norm and its dual.
def objective(x, lbda):
  return lbda*np.linalg.norm(x, ord=1) + 0.5*np.linalg.norm(x, ord=2)**2

def dual_objective(x, lbda):
  return 0.5*np.linalg.norm(soft_skrinkage(x, lbda), ord=2)**2