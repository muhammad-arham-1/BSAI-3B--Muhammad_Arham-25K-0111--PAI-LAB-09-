import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x,y,lr,epochs):
    x=x.flatten()
    y =y.flatten()
    m=0
    c=0
    n =len(x)
    losses=[]
    for i in range(1,epochs+1):
        y_hat= m*x+c
        dm = (-2/n)*np.sum(x*(y-y_hat))
        dc= (-2/n)*np.sum(y-y_hat)
        m= m-lr*dm
        c= c-lr*dc
        y_hat= m*x+c
        mse = np.mean((y-y_hat)**2)
        losses.append(mse)

        if i%100 ==0:
            print(f"Epoch {i}, MSE: {mse:.4f}")
    return m,c,losses



hours = np.array([[1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5], [5.0], [6.0], [7.0]])
marks = np.array([32, 40, 41, 50, 54, 60, 62, 70, 78, 88])

learningRate = [0.001,0.01,0.05] # 0.05 will be the fastest learning rate and 0.001 is really slow so eopchs has to be increased for it to get an appropriate output
epochs=1000
for lr in learningRate:
    print(f"Learning rate: {lr}")
    m,c,losses = gradient_descent(hours, marks,lr, epochs)
    print(f"Slope: {m:.4f}")
    print(f"Intercept: {c:.4f}")
    plt.plot(range(1,epochs+1), losses, label=lr)
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title(f"Loss Curve (Learning Rate {lr})")
    plt.show()


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(hours, marks)

print("Sklearn result: ")
print(f"Slope: {model.coef_}")
print(f"intercept: {model.intercept_}")