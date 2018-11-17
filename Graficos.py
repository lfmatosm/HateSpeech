import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

x = np.linspace(0, 2)

figure(num=None, figsize=(8, 6))
shape={2,4,6,5,4,3,1}


plt.plot([1, 2, 3, 4], [1, 4, 9, 16],'.',label='Logistic Regression',linestyle='-')# Recebe dois arrays, um sera do K e outro dos resultados
plt.plot(x, x**2, '.', label='Naïve Bayes',linestyle='-')
plt.plot(x, x**3, '.', label='AdaBoost',linestyle='-')

plt.ylim(0,1)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title("Resultados")


plt.legend()
plt.savefig('teste.png')
plt.show()