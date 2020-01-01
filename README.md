# rock-paper-scissors
https://youtu.be/0uSA3xyXlwM 

An AI to play the Rock Paper Scissors game

## Requirements
- Python 3
- Keras
- Tensorflow
- OpenCV

## Set up instructions
1. Clone the repo.
```sh
$ git clone https://github.com/SouravJohar/rock-paper-scissors.git
$ cd rock-paper-scissors
```

2. Install the dependencies
```sh
$ pip install -r requirements.txt
```

3. Gather Images for each gesture (rock, paper and scissors and None):
```sh
$ python3 gather_images.py <gesture> <image_count>
```

4. Train the model
```sh
$ python3 train.py <epochs> <model_name>
```

5. Test the model on some images
```sh
$ python3 test.py <path_to_test_image>
```

6. Play the game with your computer!
```sh
$ python3 play.py <model_name>
```
