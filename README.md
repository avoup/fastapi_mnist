Train Model:
```bash
python ./tf/train_mnist.py
```

Run api: 
```bash
uvicorn main:app
```
go to 
[swagger](http://127.0.0.1:8000/docs#/default/predict_predict__post)
<p>Upload 1:1 ratio image</p>