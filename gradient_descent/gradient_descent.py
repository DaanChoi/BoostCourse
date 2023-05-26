import sympy as sym # 기호 기반 수학 라이브러리
from sympy.abc import x

import numpy as np

def func(val):
    fun = sym.poly(x**2 + 2*x + 3)
    return fun.subs(x, val), fun
# subs: 대입

def func_gradient(fun, val):
    _, function = fun(val)
    diff = sym.diff(function, x)
    return diff.subs(x, val), diff
# diff: 미분
# '_': 무시하는 값, EX) a, _, b = (1, 2, 3) => a: 1 / b: 3

def gradient_descent(fun, init_point, lr_rate=1e-2, epsilon=1e-5): # lr_rate: 학습률(10의 -2승), epsilon: 알고리즘 종료조건(10의 -4승)
    cnt = 0
    val = init_point
    diff, _ = func_gradient(fun, val)
    while np.abs(diff) > epsilon: # 미분값의 절대값이 epsilon과 같거나 epsilon보다 작을 때 종료
        val = val - lr_rate * diff
        diff, _ = func_gradient(fun, val)
        cnt += 1
    
    print("함수: {}, 연산횟수: {}, 최소점: ({}, {})".format(fun(val)[1], cnt, val, fun(val)[0]))

gradient_descent(fun=func, init_point=np.random.uniform(-2,2)) 
# np.random.uniform(low, high, size): 균등분포로부터 무작위 표본 추출
