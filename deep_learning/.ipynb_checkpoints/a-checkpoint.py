import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def solution_model():
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-4.0, 1.0, 6.0, 11.0, 16.0, 21.0], dtype=float)
    model=Sequential()
    model.add(Dense(10,activation='relu',input_shape=[1]))
    model.add(Dense(1,activation='linear'))
    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(xs,ys,epochs=500)
    print(model.predict([10.0]))
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")