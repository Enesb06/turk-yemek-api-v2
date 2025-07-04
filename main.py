from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import tensorflow as tf
from transformers import TFAutoModelForImageClassification
import io
import numpy as np

# 1. FastAPI uygulamasını başlat
app = FastAPI(title="Türk Yemeği Tanıma API V1")

# 2. Hugging Face'den modeli yükle
# !!! DİKKAT: BURAYI KENDİ KULLANICI ADINIZ VE MODEL ADINIZLA DEĞİŞTİRİN !!!
HUGGINGFACE_MODEL_ID = "Enesb06/turk-yemegi-tanima-v2"

print(f"Model yükleniyor: {HUGGINGFACE_MODEL_ID} (Bu işlem ilk başta birkaç dakika sürebilir...)")
try:
    # Hugging Face transformers kütüphanesi, modelimizi ve config.json dosyamızı otomatik olarak çeker.
    model = TFAutoModelForImageClassification.from_pretrained(HUGGINGFACE_MODEL_ID)
    print("Model başarıyla yüklendi.")
except Exception as e:
    print(f"Model yüklenirken HATA oluştu: {e}")
    model = None

@app.get("/")
def read_root():
    return {"durum": "API Çalışıyor", "model_id": HUGGINGFACE_MODEL_ID}

@app.post("/tahmin-et")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi veya yüklenemedi.")

    # 3. Yüklenen dosyayı oku
    image_bytes = await file.read()
    
    try:
        # 4. Gelen veriyi PIL Image formatına çevir
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Geçersiz görsel dosyası: {e}")

    # 5. Modeli eğitirken kullandığımız boyuta getir (224x224)
    image = image.resize((224, 224))
    
    # 6. Görseli NumPy array'e ve ardından TensorFlow tensor formatına çevir
    image_array = np.array(image)
    # MobileNetV2 ön işlemesini uygula (Pikselleri -1 ile 1 arasına getir)
    # Eğitim sırasında bunu modelin ilk katmanı olarak yapmıştık, burada manuel yapıyoruz.
    preprocessed_image = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    
    # Batch boyutu ekle (1, 224, 224, 3)
    input_tensor = tf.expand_dims(preprocessed_image, axis=0)
    
    # 7. Model ile tahmini yap
    # Transformers kütüphanesi model çıktısını bir obje içinde verir.
    outputs = model(input_tensor)
    logits = outputs.logits
    
    # 8. En yüksek olasılıklı sınıfın index'ini bul
    predicted_class_id = int(tf.argmax(logits, axis=-1)[0])
    
    # 9. Modelin config.json'dan aldığı `id2label` sözlüğünü kullanarak yemek adını bul
    yemek_adi = model.config.id2label[predicted_class_id]
    
    # 10. Sonucu JSON olarak döndür
    return {"yemek_adi": yemek_adi}