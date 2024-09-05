#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import tensorflow as tf
print(tf.__version__)


# In[3]:


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, CuDNNGRU
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.optimizers import Adam


# In[5]:


get_ipython().system('unzip "archive (3).zip"')


# In[7]:


tweets = pd.read_csv('dataset.csv')


# In[8]:


tweets_data = tweets['Tweets']


# In[9]:


label_data = tweets['Class']


# In[10]:


tweets_data[0]


# In[11]:


label_data[0]


# In[12]:


target = tweets['Class'].values.tolist()
data = tweets['Tweets'].values.tolist()


# In[13]:


len(target)


# In[14]:


target[0]


# In[15]:


len(data)


# In[16]:


data[0]


# In[17]:


cutoff = int(len(data) * 0.80) #listeyi böldük
x_train, x_test = data[:cutoff], data[cutoff:]
y_train, y_test = target[:cutoff], target[cutoff:]


# In[18]:


x_train[299]


# In[19]:


y_train[299]


# In[20]:


num_words = 500 # kelime haznemizdeki max kelime sayısı
tokenizer = Tokenizer(num_words=num_words)


# In[21]:


type(data)


# In[22]:


data = np.array(data)


# In[23]:


data


# In[24]:


tokenizer.fit_on_texts(data)


# In[25]:


tokenizer.word_index


# In[26]:


x_train_tokens = tokenizer.texts_to_sequences(x_train)


# In[27]:


x_train[10]


# In[28]:


print(x_train_tokens[10])


# In[29]:


x_test = np.array(x_test)


# In[30]:


x_test_tokens = tokenizer.texts_to_sequences(x_test)


# In[31]:


num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)


# In[32]:


np.mean(num_tokens)


# In[33]:


np.max(num_tokens)


# In[34]:


np.argmax(num_tokens)


# In[35]:


x_train[21]


# In[36]:


max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
max_tokens


# In[37]:


np.sum(num_tokens < max_tokens) / len(num_tokens)


# In[38]:


x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens)


# In[39]:


x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens)


# In[40]:


x_train_pad.shape


# In[41]:


x_test_pad.shape


# In[42]:


np.array(x_train_tokens[10])


# In[43]:


x_train_pad[10]


# In[44]:


idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys()))


# In[45]:


def tokens_to_string(tokens):
    words = [inverse_map[token] for token in tokens if token!=0]
    text = ' '.join(words)
    return text


# In[46]:


x_train[10]


# In[47]:


tokens_to_string(x_train_tokens[10])


# In[48]:


x_train_pad[0]


# In[49]:


model = Sequential()


# In[50]:


embedding_size = 100


# In[51]:


model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='embedding_layer'))


# In[52]:


model.add(GRU(units=16, return_sequences=True))
model.add(GRU(units=8, return_sequences=True))
model.add(GRU(units=4))
model.add(Dense(1, activation='sigmoid'))


# In[53]:


optimizer = Adam(learning_rate=1e-3)


# In[54]:


from keras.losses import binary_crossentropy
lss = binary_crossentropy


# In[55]:


model.compile(optimizer= 'adam' , loss= lss, metrics=['accuracy'])


# In[56]:


model.summary()


# In[57]:


y_train = np.array(y_train)
x_train = np.array(x_train)


# In[58]:


type(x_train)


# In[59]:


x_train[0]


# In[60]:


model.fit(x_train_pad, y_train, epochs=5, batch_size=32)


# In[61]:


y_test = np.array(y_test)


# In[62]:


result = model.evaluate(x_test_pad, y_test)


# In[63]:


result[1]


# In[64]:


y_pred = model.predict(x=x_test_pad[0:1000])
y_pred = y_pred.T[0]# stünları satırlara çeviriyoruz


# In[65]:


cls_pred = np.array([1.0 if p>0.5 else 0.0 for p in y_pred])


# In[66]:


cls_true = np.array(y_test[0:1000])


# In[67]:


incorrect = np.where(cls_pred != cls_true)
incorrect = incorrect[0]


# In[68]:


len(incorrect)


# In[69]:


idx = incorrect[0]
idx


# In[70]:


text = x_test[idx]
text


# In[71]:


y_pred[idx]


# In[72]:


cls_true[idx]


# In[73]:


text1 = "Kahramanmaraş türkoğlu ilçesi şekeroba köyü çağrı sokak no 4 çadır yatak ısıtıcı ölen insanlae için de kefen ihtiyacı var."
text2 = "binalar yıkıldı su, çadır ve yemeğe ihtiyaç var yardım edin"
text3 = "bu gün günlerden pazartesi ve çok yorgunum berikadan nefret ediyorum malın teki"

texts = [text1, text2, text3]


# In[74]:


tokens = tokenizer.texts_to_sequences(texts)


# In[75]:


tokens_pad = pad_sequences(tokens, maxlen=max_tokens)
tokens_pad.shape


# In[76]:


arr = model.predict(tokens_pad)


# In[77]:


arr


# In[1]:


for a in arr:
  if(a>0.70):
    print("yardım çağrısı içeren bir cümle")
  elif(a<0.70):
    print("alakasız bir cümle")


# In[79]:


model.save('help.h5')


# In[ ]:


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By


# In[ ]:


page="https://twitter.com/i/flow/login"
USER = "username"
PASSWORD = "password"


# In[ ]:


while True:
    try:
        driver = webdriver.Firefox()
        driver.get(page)
        time.sleep(15)

        driver.find_element(By.XPATH, '/html/body/div/div/div/div[1]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/div[5]/label/div/div[2]/div/input').click
        username_input = driver.find_element(By.XPATH, '/html/body/div/div/div/div[1]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/div[5]/label/div/div[2]/div/input')
        username_input.send_keys(USER)                 #"/html/body/div/div/div/div[1]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/div[5]/label/div/div[2]/div/input"
        time.sleep(4)
        login_button = driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div[1]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/div[6]')
        login_button.click()


        time.sleep(5)

        password_input = driver.find_element(By.XPATH, '/html/body/div/div/div/div[1]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[1]/div/div/div[3]/div/label/div/div[2]/div[1]/input')
        password_input.send_keys(PASSWORD)
        time.sleep(4)
        # find the element for login and click on it
        login_button = driver.find_element(By.XPATH, '/html/body/div/div/div/div[1]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[2]/div/div[1]/div/div/div')
        login_button.click()                         #"/html/body/div/div/div/div[1]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/div[6]/div"

        hastag = "deprem"
        hastag_url = f"https://twitter.com/search?q=%23{hastag}&src=typed_query&f=live"
        #print(hastag_url)
        time.sleep(5)
        driver.get(hastag_url)
        time.sleep(5)
        run_timer = 0
        temp_text = " "
        tweet = ""
        while True:
            run_timer += 1
            for scrool in range(3):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)
            time.sleep(5)
            for i in range(10):
                try:
                    tweet = driver.find_element(By.XPATH, f'/html/body/div[1]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/section/div/div/div[{i}]/div/div/article/div/div/div[2]/div[2]/div[2]/div/span[1]').text            # "/html/body/div[1]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/section/div/div/div[4]/div/div/article/div/div/div[2]/div[2]/div[2]/div/span[1]"
                    tokens = tokenizer.texts_to_sequences(texts)
                    tokens_pad = pad_sequences(tokens, maxlen=max_tokens)
                    #tokens_pad.shape
                    arr = model.predict(tokens_pad)
                    for a in arr:
                        if(a>0.70):
                            print("Help")
                        elif(a<0.70):
                            pass
                    print("- ",tweet)

                except:
                    pass
            if temp_text == tweet:
                break
            temp_text = tweet

    except KeyboardInterrupt:
        break
        sys.exit(0)
    except:
        pass


# In[ ]:




