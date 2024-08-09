import math
import matplotlib.pyplot as plt 

def gradient_descent(dfdx, x, learning_rate=0.1, num_iterations=100):
    for iteration in range(num_iterations):
        gradient = dfdx(x)
        x = x - learning_rate * gradient
        # Optional: Print the progress to observe convergence
        print(f"Iteration {iteration+1}: x = {x:.6f}, gradient = {gradient:.6f}")
    return x

# Define the derivative function
def dfdx(x):
    return math.exp(x) - 1/x

# Initial value of x
x_initial = 0.02

# Perform gradient descent
x_min = gradient_descent(dfdx, x_initial, learning_rate=0.01, num_iterations=1000)
print(f"The value of x at minimum is approximately {x_min:.6f}")

# Perform gradient descent
x_values = gradient_descent(dfdx, x_initial, learning_rate=0.01, num_iterations=100)

# Generate the plot
plt.figure(figsize=(10, 6))
plt.plot(x_values, label='Gradient Descent Path')
plt.title('Gradient Descent Path for Minimizing f\'(x) = exp(x) - 1/x')
plt.xlabel('Iteration')
plt.ylabel('x Value')
plt.grid(True)
plt.legend()
plt.show()