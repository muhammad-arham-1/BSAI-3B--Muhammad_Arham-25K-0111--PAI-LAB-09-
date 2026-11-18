import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

hours = np.array([[1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5], [5.0], [6.0], [7.0]])
marks = np.array([32, 40, 41, 50, 54, 60, 62, 70, 78, 88])

model = LinearRegression()
model.fit(hours,marks)
print(f"Slope: {model.coef_}")
print(f"Intercept: {model.coef_}")
print(f"Marks for student studying 5.5 hours: {model.predict([[5.5]])}")
print(f"Marks for student studying 8 hours: {model.predict([[8]])}")

marks_hat= model.predict(hours)
mse= mean_squared_error(marks, model.predict(hours))
mae = mean_absolute_error(marks, model.predict(hours))
r2= r2_score(marks, model.predict(hours))
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

plt.title("Hours Studied VS Marks Obtained")
plt.scatter(hours, marks, marker='x', c='r')
plt.plot(hours, marks_hat)
plt.xlabel("Hours Studied")
plt.ylabel("Marks Achieved")
plt.show()