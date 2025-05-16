# 📊 IMDb Film Yorumları Üzerinde RNN ile Duygu Analizi

Bu projede, Keras kullanarak **RNN (Recurrent Neural Network)** modeli ile IMDb veri setindeki film yorumlarının **olumlu** veya **olumsuz** olup olmadığını sınıflandırdım. 
Hiperparametre optimizasyonu için **Keras Tuner** ile **RandomSearch** yöntemi kullandım ve modelin başarımını **AUC**, **ROC eğrisi** ve **classification report** ile analiz ettim.


# 🧠 Kullanılan Yöntemler ve Teknolojiler
- Python 🐍

- TensorFlow / Keras

- Recurrent Neural Network (RNN)

- Hyperparameter Tuning (Keras Tuner)

- ROC Curve, AUC, Accuracy

- IMDb Dataset (Keras içinden otomatik gelir)

- Matplotlib, Scikit-learn

# 🔧 Adım Adım Süreç
1. Gerekli Kütüphanelerin Yüklenmesi
```bash
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, roc_curve, auc

import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import kerastuner as kt
from kerastuner.tuners import RandomSearch
```

2. Veri Setinin Yüklenmesi ve Hazırlanması
```bash
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# Yorumları aynı uzunluğa getiriyoruz (padding)
maxlen = 100
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
```
3. Model Oluşturma Fonksiyonu (Keras Tuner ile)
```bash
def build_model(hp):
    model = Sequential()
    model.add(Embedding(input_dim=10000,
                        output_dim=hp.Int("embedding_output", 32, 128, step=32),
                        input_length=maxlen))
    model.add(SimpleRNN(units=hp.Int("rnn_units", 32, 256, step=32)))
    model.add(Dropout(rate=hp.Float("dropout_rate", 0.2, 0.5, step=0.1)))
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(optimizer=hp.Choice("optimizer", ["adam", "rmsprop"]),
                  loss="binary_crossentropy",
                  metrics=["accuracy", "AUC"])
    return model
```

4. Hiperparametre Arama (Random Search)
```bash
tuner = RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=4,
    executions_per_trial=1,
    directory="/RNN/rnn_tuner_directory",
    project_name="imdb_rnn"
)

early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

tuner.search(x_train, y_train, epochs=10, validation_split=0.2, callbacks=[early_stopping])
```
5. En İyi Modelin Değerlendirilmesi

```bash
best_model = tuner.get_best_models(num_models=1)[0]

loss, accuracy, auc_score = best_model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.3f}, Test Accuracy: {accuracy:.3f}, Test AUC: {auc_score:.3f}")
```
6. Tahmin ve Raporlama
```bash
y_pred_prob = best_model.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

print(classification_report(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
print(f"ROC AUC: {roc_auc:.3f}")
```
7. ROC Eğrisi Görselleştirmesi
```bash
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC Curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="blue", lw=2, linestyle="--")
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()
```
# 📌 Sonuç
Bu proje ile:

- RNN modellerinin doğal dil işleme görevlerinde nasıl kullanıldığını gördüm.

- Hiperparametre optimizasyonu sayesinde en iyi modeli sistematik olarak seçtim.

- Modelin başarımını hem metriklerle hem görselleştirme ile değerlendirdim.





## 👨‍💻 Geliştirici

**Teymur Mammadov** 
