# coding=utf-8

# write code...
import numpy as np
import math


###test

def klf(T, y, x0, Sigma0, F, G, Q, H, R):
    def estimate_state(F, x):
        return F.dot(x)

    def estimate_statecov(F, G, Q, Sigma):
        return F.dot(Sigma).dot(F.T) + G.dot(Q).dot(G.T)

    def get_obsereved_error(y, H, x_):
        return y[i + 1] - H.dot(x_)

    def get_observed_errorcov(H, Sigma_, R):
        return H.dot(Sigma_).dot(H.T) + R

    def get_kalmangain(Sigma_, H, S):
        if S.ndim == 0:
            K = Sigma_.dot(H.T) * 1 / S
        else:
            K = Sigma_.dot(H.T).dot(np.linalg.inv(S))
        return K

    def update_state(x_, K, ei):
        return x_ + K.dot(ei)

    def update_error_mat(Sigma_, K, H):
        temp_mat = K.dot(H)
        if temp_mat.ndim == 0:
            Sigma = Sigma_ - temp_mat * Sigma_
        else:
            Sigma = Sigma_ - temp_mat.dot(Sigma_)
        return Sigma

    def get_estimated_y(H, x):
        return H.dot(x)


    #Initial values
    x = x0
    Sigma = Sigma0

    #Initial values of output of this function
    M = x
    x_estimated = x[0].T
    y_estimated = [H.dot(x)]
    ei_values = [0]
    S_values = [R]


    for i in range(T):
        #estimate
        x_ = estimate_state(F, x)
        Sigma_ = estimate_statecov(F, G, Q, Sigma)

        if math.isnan(y[i+1]):
            x = x_
            Sigma = Sigma_
            y_ = get_estimated_y(H, x)

        else:
            #update
            ei = get_obsereved_error(y, H, x_) # 観測残差
            S = get_observed_errorcov(H, Sigma_, R)
            K = get_kalmangain(Sigma_, H, S)
            x = update_state(x_, K, ei)
            Sigma =  update_error_mat(Sigma_, K, H)
            y_ = get_estimated_y(H,x)

        M = np.c_[M, x]
        x_estimated = np.r_[x_estimated, x[0].T]
        y_estimated.append(y_)
        ei_values.append(ei)
        S_values.append(S)

    M = M.T

    return M, x_estimated, y_estimated, ei_values, S_values


def get_klf_prediction(x, F, H):
    x_ = F.dot(x)
    y_ = H.dot(x_)
    return x_, y_
