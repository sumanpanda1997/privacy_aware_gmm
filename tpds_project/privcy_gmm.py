
# coding: utf-8

# In[4]:


import numpy as np


# In[5]:


'''this is code for testing privacy in GMM 
dataset taken from Gowala social networking site'''
new_f = open("new.txt","r")
fl_new = new_f.readlines()
lst = []
for i in fl_new:
    l1 = i.split()
    l1 = list(map(float,l1[2:4]))
    lst.append(l1)
new_f.close()
print(len(lst))


# In[6]:


X = np.array(lst)


# In[7]:


l = len(X)
idx = 4*l//5


# In[8]:


X_train = X[:idx]
X_test = X[idx:]


# In[9]:


X_test


# In[10]:


X_new = norm_data(X_test)


# In[11]:


X_new


# In[38]:


def pnorm(x, m, s):
    """ 
    Compute the multivariate normal distribution with values vector x,
    mean vector m, sigma (variances/covariances) matrix s
    """
    xmt = np.matrix(x-m).transpose()
    for i in range(len(s)):
        if s[i,i] <= sys.float_info[3]: # min float
            s[i,i] = sys.float_info[3]
    try:
        sinv = np.linalg.inv(s)
        xm = np.matrix(x-m)
        a = (2.0*math.pi)**(-len(x)/2.0)*(1.0/math.sqrt(np.linalg.det(s)))*math.exp(-0.5*(xm*sinv*xmt))
    except:
        a = random.uniform(0, 1)
    return a


# In[47]:


'''this function finds the initial guess for mean, sigma and theta.If we want to change it furthe then we can chage it'''

def initpara(n,k,d):
    
    mean = np.zeros((k,d),float)
    sigma = np.zeros((k,d,d),float)

    for i in range(int(k)):
        sigma[i] = np.eye(d,dtype = float)

    theta = numpy.random.dirichlet([1/k]*k)
    return mean,sigma,theta


# In[48]:


'''this is the function to find the membership_weight matrix using the parameter theta of the gaussian models 
and the parameter of the distribution of the latent variable Z
the calculated matrix will be used to update the parameter in the maximization step'''


def expectation_step(data,mean,sigma,theta,n,k):
    membership_weight = np.zeros((n,k),float)
    count = 0
    log_likelihood = 0
    for i in range(n):
#         print('data is',data[i])
        deno = 0
        for j in range(k):
            
#             print('kluster is', j,'mean is ',mean[j],'sigma is\n',(sigma[j]))
            prob = (pnorm(data[i], mean[j], sigma[j]))*theta[j]
            deno = deno+prob
            membership_weight[i][j] = prob
        
        try:
            a = math.log(deno)
            b = membership_weight[i]/deno
            log_likelihood += a
            membership_weight[i] = b
        except:
            membership_weight[i] = numpy.random.dirichlet([1/k]*k)
    

            
#     print('\n\n\n\n--------+++++++++end expectatipn++++++++-------------\n\n\n\n')
#     print(membership_weight)
    return membership_weight,log_likelihood


# In[49]:


'''this function uses the weight matrix found in the expectation step to find the updated parameters. 
this is known as maximization step.
Here we are going to find the new mean,variance and theta parameter of the GMM'''

def maximization_step(weight,data,mean,sigma,theta,n,k):
    
    effective_cluster_point = np.zeros(k,float)
    mean_weight = 0
    for j in range(k):
        mean_weight = 0
        for i in range(n):
            effective_cluster_point[j] += weight[i][j]
            mean_weight += data[i]*weight[i][j]
            
        theta[j] = effective_cluster_point[j]/n
        mean[j] = mean_weight/effective_cluster_point[j]
    
    for j in range(k):
        num_ = 0
        for i in range(n):
            a = data[i] - mean[j]
            a = np.outer(a,a.transpose())
            num_ += weight[i][j]*a
            
        sigma[j] = num_/effective_cluster_point[j]
        
#     print('\n\n\n\n\--------+++++++++end amximization++++++++-------------\n\n\n\n')
#     print('mean is\n',mean,'sigma is \n',sigma,'theta is \n',theta)
    return mean,sigma,theta


# In[51]:


'''this function checks the convergence criteria for the GMM EM algorithm if not then calls EM iteratively'''

def checking(data,n,d,k):
#     print('k is in checking----',k)
    log_likelihood = 0
    epsilon = 0.0001
    mean,sigma,theta = initpara(n,k,d)
#     print(sigma)
    max_lim = 100
    i = 0
    while i < max_lim:        
        membership_weight,log_likelihood_ = expectation_step(data,mean,sigma,theta,n,k)
        mean_,sigma_,theta_ = maximization_step(membership_weight,data,mean,sigma,theta,n,k)
        
        mean_noise,sigma_noise,theta_noise = add_laplacian_noise(mean_,sigma_,theta_,n,)
        
        mean = mean_
        sigma = sigma_
        theta = theta_
        threshold = log_likelihood_ - log_likelihood
        log_likelihood = log_likelihood_
        
        
        if threshold < 0:
            threshold = -threshold
        if threshold < epsilon:
            break
        i += 1


    return mean,sigma,theta,membership_weight
    


# In[1]:


def norm_data(x_ls):
    n,_ = x_ls.shape
    for i in range(n):
        s = sum(abs(x_ls[i]))
        x_ls[i] = x_ls[i]/s
    return x_ls


# In[12]:


import math


# In[13]:


def add_laplacian_noise(mean,sigma,theta,n,d,eps,delta):
    
    #adding laplacian noise in theta
    theta_sensitivity = 2/n
    sz1 = np.shape(theta)
    b1 = theta_sensitivity/eps
    
    theta_lap_noise = np.random.laplace(b1,size = (sz1,1))
    theta_pur = theta+theta_noise
    
    #adding gaussian noise in theta
    b1_gaussian = (theta_sensitivity**2)/(eps**2)
    theta_gaus_sigma = 2*(math.log(1.25/delta))* b1_gaussian
    
    theta_gaus_noise = np.random.normal(0, theta_gaus_sigma, sz1)
    theta_gaus_pur = theta+theta_gaus_noise
    
    

    #adding noise both in k means and k variances for each k
    sz2 = np.shape(mean[0])
    mean_pur = np.zeros(sz1,sz2)
    mean_gaussain_pur = np.zeros(sz1,sz2)
    cov_pur = np.zeros(k,d,d)
    cov_gaussian = np.zeros(k,d,d)
    
    for k in range(sz1):
        n_mod = n*theta_pur[k]
        
        mean_sensitivity_k = 2*(math.sqrt(d))/n_mod
        b2 = mean_sensitivity_k/eps
        
        mean_noise_k = np.random.laplace(b2,(sz2,1))
        mean_pur[k] = mean[k]+mean_noise_k
        
        mean_gaus_sigma = 
        
        sigma_sesitivity_k =  (2/n_mod)
        b3 = (sigma_sesitivity_k**2)/(eps**2)
        
        beta = 2*(math.log(1.25/delta))* b3
        cov = beta*np.eye(d*(d+1)/2)
        z = np.random.multivariate_normal(0, cov)
        cov_Z = np.zeros(d,d)
        k = 0
        for i in range(d):
            for j in range(i,d):
                if i==j:
                    cov_Z[i][i] = z[k]
                cov_Z[i][j] = z[k]
                cov_Z[j][i] = z[k]
                k += 1
        cov_noise_k = sigma[k]+cov_Z
        cov_PSD = nearPSD(cov_noise_k)
        cov_pur[k] = cov_PSD
        
    return theta_pur,mean_pur,cov_pur
        
    
    
    


# In[14]:


def nearPSD(A,epsilon=0):
    n = A.shape[0]
    eigval, eigvec = np.linalg.eig(A)
    val = np.matrix(np.maximum(eigval,epsilon))
    vec = np.matrix(eigvec)
    T = 1/(np.multiply(vec,vec) * val.T)
    T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
    B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
    out = B*B.T
    return(out)


# In[ ]:


def add_gaussian_noise(mean,sigma,theta,n,d,eps,delta):
    

