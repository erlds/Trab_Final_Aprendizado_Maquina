#Aprendizado de Máquina
#Alunos: Juan e Evaristo
#Professor:Daniel Guerreiro 

import matplotlib.pyplot as plt
import pandas as pd

df1=pd.read_csv("run1/record.txt")
#print(df1)

#plt.plot(df1.loc[:,'epoch'], df1.ix[:,-2])
plt.plot(df1.loc[:,'epoch'], df1.ix[:,-1])
plt.xlabel('epoch')
plt.ylabel('correct test values')
#plt.title('About as simple as it gets, folks')
plt.grid(True)
plt.savefig("test.png")
plt.show()