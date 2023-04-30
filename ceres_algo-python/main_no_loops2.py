from qr_decomposition import *
import numpy as np
import scipy.linalg
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

def levenberg_marquardt(r, J, x, Delta = 100, Delta_max = 10000,
                        eta = 0.0001, sigma = 0.1,
                        nmax = 500, tol_abs = 10**(-7),
                        tol_rel = 10**(-7), eps = 10**(-3),
                        Scaling = False):
    def f(x):
        return 0.5*np.linalg.norm(r(x))**2
    def gradf(x):
        return np.dot(np.transpose(J(x)),r(x))
    counter = 0
    fx = f(x)
    func_eval = 1
    m,n = J(x).shape
    
    Information = [['counter', 'norm of step p',
                    'x', 'norm of the gradient at x']]
    
    Information.append([counter, 'not available', x,
                        np.linalg.norm(gradf(x))])#(x)
    
    tolerance = min((tol_rel * Information[-1][-1] + tol_abs), eps)
    
    D = np.eye(n)
    D_inv = np.eye(n)
    
    while Information[-1][-1] > tolerance and counter < nmax:
        Jx = J(x)
        if Scaling == True:
            # for i in list(range(0, n, 1)):
            #     D[i,i] = max(D[i,i], np.linalg.norm(Jx[:,i]))
            # #объединить
            # for i in list(range(0, n, 1)):
            #     D_inv[i,i] = 1/D[i,i]
            D = np.eye(n) * np.amax(np.abs(Jx), axis=0)
            D_inv = np.linalg.inv(D)

        D_2 = np.dot(D, D)
        Q, R, Pi = scipy.linalg.qr(Jx, pivoting=True)
        P = np.eye(n)[:,Pi]
        rank = np.linalg.matrix_rank(Jx)
        if rank == n:
            p = np.dot(P, scipy.linalg.solve_triangular(
                R[0:n,:], np.dot(Q[:,0:n].T, -r(x))))
        else:
            y = np.zeros(n)
            y[0:rank] = scipy.linalg.solve_triangular(
                R[0:rank,0:rank], np.dot(Q[:,0:rank].T, -r(x)))
            p = np.dot(P, y)
        
        Dp = np.linalg.norm(np.dot(D, p))
        
        if Dp <= ((1+sigma) * Delta):
            alpha = 0
        else:
            J_scaled = np.dot(Jx, D_inv)
            u = np.linalg.norm(np.dot(J_scaled.T, r(x))) / Delta
            if rank == n:
                q = scipy.linalg.solve_triangular(
                    R[0:n,:].T, np.dot(P.T, np.dot(D_2, p)),
                    lower = True)
                l = (Dp - Delta) / (np.linalg.norm(q)**2 / Dp)
            else:
                l = 0
            
            if u == np.inf:
                alpha = 1
            else:
                alpha = max(0.001 * u, (l * u)**(0.5))
            
            while Dp > (1 + sigma) * Delta or Dp < (1 - sigma) * Delta:
                if alpha == np.inf:
                    print('Error: '
                          + 'The LM method fails to converge.'
                          + '(Lambda gets too large)'
                          + 'Please try a different starting point.')
                    return x, Information
                if alpha <= l or alpha > u:
                    alpha = max(0.001 * u, (l * u)**(0.5))
                
                D_lambda = np.dot(P.T, np.dot(D, P))
                R_I = np.concatenate((R, alpha**(0.5) * D_lambda), axis = 0)
                
                #decomposition usage
                R_lambda, Q_lambda2 = givens_qr(R_I, n, m)
                
                Q_lambda = np.dot(np.concatenate(
                    (np.concatenate((Q, np.zeros((m,n))), axis = 1),
                     np.concatenate((np.zeros((n,m)), P), axis = 1)),
                    axis = 0), Q_lambda2)
                
                r_0 = np.append(r(x), np.zeros(n))
                
                p = np.dot(P, scipy.linalg.solve_triangular(
                    R_lambda[0:n,:], np.dot(Q_lambda[:,0:n].T, -r_0)))
                
                Dp = np.linalg.norm(np.dot(D, p))
                
                q = scipy.linalg.solve_triangular(
                    R_lambda[0:n,:].T,
                    np.dot(P.T, np.dot(D_2, p)), lower = True)
                
                phi = Dp - Delta
                phi_derivative = -np.linalg.norm(q)**2 / Dp
                
                if phi < 0:
                    u = alpha
                l = max(l, alpha - phi / phi_derivative)
                alpha = alpha - ((phi + Delta) / Delta) * (phi / phi_derivative)
        
        fxp = f(x + p) 
        func_eval += 1
        if fxp > fx or fxp == np.inf or np.isnan(fxp) == True:
            rho = 0
        else:
            ared = 1 - (fxp / fx)
            pred = (0.5 * np.linalg.norm(np.dot(Jx, p))**2) / fx + (alpha * Dp**2) / fx
            rho = ared / pred
        
        if rho < 0.25:
            Delta = 0.25 * Delta
        else:
            if rho > 0.75 and Dp >= (1 - sigma) * Delta:
                Delta = min(2 * Delta, Delta_max)
            # else:
            #     Delta = Delta
        if rho > eta:
            x += p
            fx = fxp
            counter += 1
            Information.append([counter, np.linalg.norm(p), x, 
                                np.linalg.norm(gradf(x))])
    
    if Information[-1][-1] <= tolerance:
        print('The LM method terminated successfully.')
        print('\n Current function value: ' + str(fx))
        print('Iterations: ' + str(counter))
        print('Function evaluations: '+ str(func_eval))
    else:
        print('The LM method fails to converge within'+ str(nmax) + 'steps.')
    return x, Information

# def r(x):
#     r = np.zeros((2,))
#     r[0] = x[0]**2 + x[1]**2 - 1
#     r[1] = x[0] - x[1]**2
#     return r

# def J(x):
#     J = np.zeros((2, 2))
#     J[0, 0] = 2*x[0]
#     J[0, 1] = 2*x[1]
#     J[1, 0] = 1
#     J[1, 1] = -2*x[1]
#     return J

def r(x):
    
    r = np.zeros((2,))
    
    _, tvec, camera, xy_obs = x
    xp = -tvec[0]/tvec[2]
    yp = -tvec[1]/tvec[2]
    r2 = xp**2 + yp**2
    l1 = 1
    l2 = 1
    distortion = 1.0 + r2*(l1 + l2*r2)
    focal = camera[0][0]
    predicted_x = focal*distortion*xp
    predicted_y = focal*distortion*yp
    
    r[0] = predicted_x - xy_obs[0]
    r[1] = predicted_y - xy_obs[1]
    
    return r
# def r(x):
#     r = np.zeros((2,))
#     r[0] = x[0]**2 + x[1]**2 - 1
#     r[1] = x[0] - x[1]**2
#     return r

# def J(x):
#     J = np.zeros((2, 2))
#     J[0, 0] = 2*x[0]
#     J[0, 1] = 2*x[1]
#     J[1, 0] = 1
#     J[1, 1] = -2*x[1]
#     return J
def J(x):
    _, tvec, camera, xy_obs = x
    xp = -tvec[0]/tvec[2]
    yp = -tvec[1]/tvec[2]
    l1 = 1
    l2 = 1
    focal = camera[0][0]
    
    distortion = 1.0 + l1*(xp**2 + yp**2) + l2*(xp**2 + yp**2)**2
    
    
    predicted_x = focal*xp + focal*xp*l1*(xp**2 + yp**2) + focal*xp*l2*(xp**2 + yp**2)**2
    predicted_y = focal*yp + focal*yp*l1*(xp**2 + yp**2) + focal*yp*l2*(xp**2 + yp**2)**2
    
    
    J = np.zeros((2, 2))
    J[0, 0] = 2*x[0]
    J[0, 1] = 2*x[1]
    J[1, 0] = 1
    J[1, 1] = -2*x[1]
    
    return J

x0 = np.array([0.97, 0.22])
x = (np.array([-39608.39506173,11111.11111111,0.]), np.array([99825.529,-115.,45000.]), np.array([[1.5e+03,0.0e+00,5.0e+02],[0.0e+00,1.5e+03,5.0e+02],[0.0e+00,0.0e+00,1.0e+00]]), np.array([4600.32,5900.24]))
# rvec, tvec, intrinsic, xy_observed

print(J(x).shape)

# x, Information = levenberg_marquardt(r, J, x0, Scaling=True)
