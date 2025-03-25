import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from matplotlib.widgets import Button, TextBox

class SineNet(nn.Module):
    def __init__(self):
        super(SineNet, self).__init__()
        # Network with 2 hidden layers
        self.hidden1 = nn.Linear(1, 4)
        self.hidden2 = nn.Linear(4, 4)
        self.output = nn.Linear(4, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.output(x)
        return x

    def get_weights_and_biases(self):
        # Returns a list of (name, numpy array) tuples.
        params = []
        for name, param in self.named_parameters():
            params.append((name, param.data.numpy()))
        return params

    def evaluate_at_x(self, x_value):
        """Evaluate the network at a specific x value"""
        with torch.no_grad():
            x = torch.FloatTensor([[x_value]])
            return self.forward(x).item()

# Generate training data
x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = np.sin(x)
x_train = torch.FloatTensor(x.reshape(-1, 1))
y_train = torch.FloatTensor(y.reshape(-1, 1))

# Initialize model, optimizer, and loss function
model = SineNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
criterion = nn.MSELoss()

# Global variables for control
paused = False
last_x_input = None

def on_pause_button(event):
    global paused
    paused = not paused
    pause_button.label.set_text('Resume' if paused else 'Pause')

def on_text_submit(text):
    global last_x_input
    try:
        x_value = float(text)
        last_x_input = x_value
        if paused:
            y_value = model.evaluate_at_x(x_value)
            ax1.text(0.02, 0.02, f'f({x_value:.2f}) = {y_value:.3f}', 
                    transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8))
            plt.draw()
    except ValueError:
        print("Please enter a valid number")

# Set up the figure for animation
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(121)  # For the sine function plot
ax2 = fig.add_subplot(122)  # For the network visualization

# Add pause button
pause_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
pause_button = Button(pause_ax, 'Pause')
pause_button.on_clicked(on_pause_button)

# Add text input for x value
text_ax = plt.axes([0.81, 0.15, 0.1, 0.075])
text_box = TextBox(text_ax, 'Enter x:', initial='0.0')
text_box.on_submit(on_text_submit)

# Plot the true sine function
#explain the following line
line_true = ax1.scatter(x, y,s=5, label='True Sine')
line_pred, = ax1.plot([], [], 'r--', label='Neural Network')
ax1.set_title('Function Approximation')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()
ax1.grid(True)

def normalize_for_viz(value, min_alpha=0.1, max_alpha=0.9):
    """Normalize weight values to a reasonable alpha range using a sigmoid."""
    #return (value - np.min(value)) / (np.max(value) - np.min(value))

    return min_alpha + (max_alpha - min_alpha) * (1 / (1 + np.exp(-value)))

def draw_neural_network(ax, weights_and_biases):
    ax.clear()
    ax.set_axis_off()
    
    # Define the network structure: [input, hidden1, hidden2, output]
    layers = [1, 4, 4, 1]
    layer_positions = [0, 1, 2, 3]
    
    # Separate weights and biases
    weights = []
    biases = []
    for name, param in weights_and_biases:
        if 'weight' in name:
            weights.append(param)
        else:
            biases.append(param)

    eqbiases = []
    eqweights = []
    
    # Move everything up by adjusting the y-position calculation
    y_offset = 1  # Add an offset to move everything up
    
    # Dictionary to store node positions for drawing connections
    node_positions = {}
    for l, layer_size in enumerate(layers):
        for n in range(layer_size):
            y_position = (layer_size - 1) / 2 - n + y_offset
            node_positions[(l, n)] = (layer_positions[l], y_position)
            circle = plt.Circle((layer_positions[l], y_position), 0.1, fill=True, color='lightblue')
            ax.add_artist(circle)
            
            if l > 0:
                bias_val = biases[l-1][n] if l < len(layers)-1 else biases[l-1][0]
                ax.text(layer_positions[l], y_position - 0.2, f'b: {bias_val:.2f}', ha='center', va='top')
            bias_val = biases[l-1][n] if l < len(layers)-1 else biases[l-1][0]
            eqbiases.append([float(bias_val), layer_size])
    
    # Draw connections with weights
    for l in range(len(layers) - 1):
        weight_matrix = weights[l]
        for i in range(layers[l]):
            for j in range(layers[l + 1]):
                start = node_positions[(l, i)]
                end = node_positions[(l + 1, j)]
                weight_val = weight_matrix[j][i] if l < len(layers) - 2 else weight_matrix[0][i]
                line_color = 'red' if weight_val < 0 else 'green'
                alpha = normalize_for_viz(abs(weight_val))
                ax.plot([start[0], end[0]], [start[1], end[1]], color=line_color, alpha=alpha)
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2
                eqweights.append([float(weight_val), layer_size])
                ax.text(mid_x, mid_y, f'{weight_val:.2f}', ha='center', va='center', fontsize=8)
    
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-3.0, 3.5)  # Adjusted y-limits to accommodate the moved network

    # Create equations for all layers
    equations = []
    
    # First layer
    for i in range(4):
        equations.append(f'h1_{i+1} = tanh({eqweights[i][0]:.3f}x + {eqbiases[i][0]:.3f})')
    
    # Second layer
    for i in range(4):
        terms = []
        for j in range(4):
            weight_idx = 4 + i*4 + j
            terms.append(f'{eqweights[weight_idx][0]:.3f}*h1_{j+1}')
        equations.append(f'h2_{i+1} = tanh({" + ".join(terms)} + {eqbiases[i+4][0]:.3f})')
    
    # Output layer
    equations.append("f(x) = ")
    final_terms = []
    for i in range(4):
        weight_idx = 20 + i
        final_terms.append(f'{eqweights[weight_idx][0]:.3f}*h2_{i+1}')
    equations.append("       " + " + ".join(final_terms))
    equations.append(f"       + {eqbiases[-1][0]:.3f}")
    
    # Add equations to plot - moved up and left
    ax.text(-0.4, -1.2, "Neural Network Equation:", ha='left', va='center', fontsize=9)
    y_start = -1.4
    for i, line in enumerate(equations):
        ax.text(-0.4, y_start - (i * 0.12), line, ha='left', va='center', fontsize=6)
    
    # Create the complete expanded equation
    combined_eq_parts = ["f(x) = "]
    for i in range(4):  # for each output weight
        weight_idx = 20 + i
        output_weight = eqweights[weight_idx][0]
        
        # Start building the term for this path through the network
        term = f"{output_weight:.3f}*tanh("
        
        # Add all connections to this second layer neuron
        second_layer_terms = []
        for j in range(4):
            weight_idx_2 = 4 + i*4 + j
            second_weight = eqweights[weight_idx_2][0]
            second_layer_terms.append(f"{second_weight:.3f}*tanh({eqweights[j][0]:.3f}x + {eqbiases[j][0]:.3f})")
        
        term += " + ".join(second_layer_terms)
        term += f" + {eqbiases[i+4][0]:.3f})"
        
        if i < 3:
            term += " + "
        combined_eq_parts.append(term)
    
    # Add final bias
    combined_eq_parts.append(f" + {eqbiases[-1][0]:.3f}")
    
    # Add the combined equation at the bottom with line breaks
    ax.text(-0.4, -2.8, "Complete Expanded Equation:", ha='left', va='center', fontsize=9)
    
    # Display equation parts on separate lines, shifted left
    y_start = -3.0
    line_height = 0.1
    for i, part in enumerate(combined_eq_parts):
        if i == 0:  # First line with f(x) =
            ax.text(-0.4, y_start, part, ha='left', va='center', fontsize=6)
        else:  # Subsequent lines should be indented slightly
            ax.text(-0.3, y_start - (i * line_height), part, ha='left', va='center', fontsize=6)
   
    ax.set_title('Network Architecture\nGreen: positive weights, Red: negative weights')

def init():
    line_pred.set_data([], [])
    return line_pred,

def animate(frame):
    global paused
    if paused:
        return line_pred,
        
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    
    # Update the prediction plot
    with torch.no_grad():
        y_pred = model(x_train)
    line_pred.set_data(x, y_pred.numpy())
    
    # Update the network architecture drawing
    draw_neural_network(ax2, model.get_weights_and_biases())
    
    # Display the function described by the neural network
    weights_and_biases = model.get_weights_and_biases()
    #print(weights_and_biases)
    #func_text = f"y = tanh({weights_and_biases[0][1][0][0]:.2f}*x + {weights_and_biases[1][1][0]:.2f})"
    #func_text += f" + tanh({weights_and_biases[2][1][0][0]:.2f}*x + {weights_and_biases[3][1][0]:.2f})"
    #func_text += f" + {weights_and_biases[4][1][0]:.2f}"
    #ax1.text(0.5, -1.5, func_text, ha='center', va='center', transform=ax1.transAxes, fontsize=12)
    
    plt.title(f'Epoch {frame+1}, Loss: {loss.item():.4f}')
    
    return line_pred,

# Create and show the animation
anim = FuncAnimation(fig, animate, init_func=init, frames=10000, interval=100, blit=False)
plt.tight_layout()
plt.show()

with torch.no_grad():
    final_loss = criterion(model(x_train), y_train)
print(f'\nFinal loss: {final_loss.item():.4f}')
