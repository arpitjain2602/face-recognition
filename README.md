# face-recognition
Implements Siamese Neural Network for One Shot Image Recognition

## Handling data
Enter your images in the folder **data**
Make sure you enter images class-wise, i.e. if you have photos of 100 different people, make 100 folders inside **data** and keep at least 1 image inside each of it.

## Running the model

Run the python files in the order
```
python create_face_embeddings
python align_dataset_mtcnn
python rest-server
```

The last commands launches a server at http://127.0.0.1:5000/ on which the FR system can be accessed

Feel free to ping me in case anything is broken. Thanks for visiting. Cheers! :simple_smile:
