#!/usr/bin/env python
# coding: utf-8
# Regressão linear, exemplificando planos de saúde por idade do paciente
# In[ ]:


import numpy as np #importamos a biblioteca numpy

# O eixo X são as idades
# In[ ]:


X = np.array([ [18], [22], [30],  [35], [46], [49], [54], [63], [77], [83] ])


# In[ ]:


X

# O eixo Y são os preços
# In[ ]:


y = np.array([ [840], [1000], [1450], [1600], [2100], [2500], [2900], [3400], [4300], [5120] ])


# In[38]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(X, y)

# Agora encontramos as variáveis b0 e b1 da fórmula da regressão linear.
# In[ ]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)


# In[ ]:


# Valores

#b0 
regressor.intercept_


# In[ ]:


#b1
regressor.coef_

# Agora nós iremos fazer as previsões dos preços com base nos nossos coeficientes. É como se pedissemos para que ele mostre o que entendeu do nosso gráfico.
# In[ ]:


previsoes = regressor.predict(X)

# Pedindo para que o algorítmo prevesse o preço do plano de saúde para as mesmas idades do array X
# In[ ]:


previsoes


# In[ ]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
mae = mean_absolute_error(y, previsoes)
mse = mean_squared_error(y, previsoes)

# O quanto o nosso algoritmo errou com base na análise de erro via MEAN ABSOLUTE ERROR
# In[44]:


mae

# O quanto o nosso algoritmo errou com base na análise de erro via MEAN SQUARED ERROR
# In[45]:


mse

# Então ao fim é elaborado um gráfico, onde os circulos em azul indicam o valor ideal da resposta, que é a resposta quando o algorítimo atinge 100% de "sabedoria", e os circulos em vermelho indicam a resposta do algoritmo após as nossas definições, ou seja, depois de "treinarmos" o algorítmo.
# In[50]:


plt.plot(X, y, 'o', color = 'blue')
plt.plot(X, previsoes, 'o', color = 'red')
plt.title('Regressão linear simples - Plano de saúde')
plt.xlabel('Idade')
plt.ylabel('Preço')

