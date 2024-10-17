import matplotlib.pyplot as plt
import numpy as np

def plot_f1_scores(methods, f1_scores_nb, f1_scores_lr):
    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, f1_scores_nb, width, label='Naive Bayes')
    rects2 = ax.bar(x + width/2, f1_scores_lr, width, label='Regresión Logística')

    ax.set_ylabel('F1 Score')
    ax.set_title('Comparación de F1 Scores por Método y Clasificador')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
