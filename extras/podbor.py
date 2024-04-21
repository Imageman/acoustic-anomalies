# try:
#     import platform
#     if float(platform.release()) > 6.1:  # Windows 7 	6.1
#         from sklearnex import patch_sklearn  # https://intel.github.io/scikit-learn-intelex/
#         patch_sklearn()  # Intel заплатка для ускорения sklearn
#     else:
#         print('Minimum version for Intel sklearnex Win8.')
# except:
#     print('For faster work sklearn install sklearnex')
#     pass


from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

from loguru import logger
import numpy as np

def ScreePlotFA(X):
    # функция должна была показывать как число компонент влияет на объяснимость
    # в реальности показывает значимость каждого входного параметра?
    max_n= 230
    fa = FactorAnalysis(n_components=max_n)
    fa.fit(X)
    # вычисление объясненной дисперсии для каждой компоненты
    explained_variance_ratio = 1 - fa.noise_variance_ / np.var(X, axis=0).sum()
    # explained_variance_ratio = fa.explained_variance_ratio_

    # создание графика
    plt.plot(np.arange(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'o-')
    plt.xlabel('Количество компонент')
    plt.ylabel('Доля объясненной дисперсии')
    plt.title('Scree plot')
    plt.show()


def proborFA(X):
    # перебираем разные варианты и смотрим score для каждого (больше - лучше)
    # ищем место, начиная с которого прибавка маленькая
    n_components = range(1,320,10)

    fa = FactorAnalysis()
    bics = []
    for n in n_components:
        fa.n_components = n
        fa.fit(X)
        bics.append(fa.score(X))
    
    plt.plot(n_components, bics, label='score')
    plt.xlabel('Number of Components')
    plt.ylabel('score')
    plt.legend()
    plt.show()

