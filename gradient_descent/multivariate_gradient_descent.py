import sympy as sym
from sympy.abc import x, y

import numpy as np

def eval_(fun, val): 
    val_x, val_y = val
    fun_eval = fun.subs(x, val_x).subs(y, val_y)
    return fun_eval
# eval 의미: 표현식의 결과값을 리턴해주는 함수
# 이름 끝에 '_': 파이썬 키워드와 충돌을 피하기 위해

def func_multi(val):
    x_, y_ = val
    func = sym.poly(x**2 + 2*y**2)
    return eval_(func, [x_, y_]), func

def func_gradient(fun, val):
    x_, y_ = val
    _, function = fun(val)
    diff_x = sym.diff(function, x) # x에 대해 미분한 도함수
    diff_y = sym.diff(function, y) # y에 대해 미분한 도함수
    grad_vec = np.array([eval_(diff_x, [x_, y_]), eval_(diff_y, [x_, y_])], dtype=float) # 다변수 벡터에 대한 미분값 벡터
    return grad_vec, [diff_x, diff_y]

def gradient_descent(fun, init_point, lr_rate=1e-2, epsilon=1e-5):
    cnt = 0
    val = init_point
    diff, _ = func_gradient(fun, val)
    while np.linalg.norm(diff) > epsilon: # 벡터이기 때문에 L2-norm 사용
        val = val - lr_rate * diff
        diff, _ = func_gradient(fun, val)
        cnt += 1
    print("함수: {}, 연산횟수: {}, 최소점: ({}, {})".format(fun(val)[1], cnt, val, fun(val)[0]))

pt = [np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
gradient_descent(fun=func_multi, init_point=pt)