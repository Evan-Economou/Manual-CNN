"""
CNN Framework for Normal vs Pneumonia Chest X-Ray Classification

"""

import numpy as np
import pickle
from pathlib import Path


class ConvLayer:
    def __init__(self, num_filters, filter_size, input_channels):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        
        #forward pass
        self.filters = np.random.randn(num_filters, input_channels, filter_size, filter_size) * np.sqrt(2.0 / (input_channels * filter_size * filter_size))
        self.biases = np.zeros(num_filters)
        
        #backward pass
        self.input = None
    
    def forward(self, input_data):
        """
        Forward pass for convolutional layer
        Args:
            input_data: shape (batch_size, channels, height, width)
        Returns:
            output: shape (batch_size, num_filters, output_h, output_w)
        """
        self.input = input_data
        batch_size, channels, h, w = input_data.shape
        
        output_h = h - self.filter_size + 1
        output_w = w - self.filter_size + 1
        
        output = np.zeros((batch_size, self.num_filters, output_h, output_w))
        
        #The Convolator
        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(output_h):
                    for j in range(output_w):
                        region = input_data[b, :, i:i+self.filter_size, j:j+self.filter_size]
                        output[b, f, i, j] = np.sum(region * self.filters[f]) + self.biases[f]
        
        return output
    
    def backward(self, d_output, learning_rate):
        batch_size, channels, h, w = self.input.shape

        #initialize arrays for gradients
        d_input = np.zeros_like(self.input)
        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        
        output_h = d_output.shape[2]
        output_w = d_output.shape[3]
        
        #compute gradients
        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(output_h):
                    for j in range(output_w):
                        region = self.input[b, :, i:i+self.filter_size, j:j+self.filter_size]
                        
                        d_filters[f] += d_output[b, f, i, j] * region
                        d_biases[f] += d_output[b, f, i, j]
                        d_input[b, :, i:i+self.filter_size, j:j+self.filter_size] += d_output[b, f, i, j] * self.filters[f]
        
        #update weights based on local gradient
        self.filters -= learning_rate * d_filters / batch_size
        self.biases -= learning_rate * d_biases / batch_size
        
        return d_input


class MaxPoolLayer:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size
        self.input = None
        self.max_indices = None
    
    def forward(self, input_data):
        """
        Forward pass for max pooling

        Args:
            input_data: shape (batch_size, channels, height, width)
        Returns:
            output: shape (batch_size, channels, output_h, output_w)
        """
        self.input = input_data
        batch_size, channels, h, w = input_data.shape
        
        output_h = h // self.pool_size
        output_w = w // self.pool_size
        
        output = np.zeros((batch_size, channels, output_h, output_w))
        self.max_indices = np.zeros((batch_size, channels, output_h, output_w, 2), dtype=int)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_h):
                    for j in range(output_w):
                        region = input_data[b, c, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size]
                        max_val = np.max(region)
                        output[b, c, i, j] = max_val
                        
                        #find indices of max value for backprop
                        max_pos = np.unravel_index(np.argmax(region), region.shape)
                        self.max_indices[b, c, i, j] = (max_pos[0] + i*self.pool_size, max_pos[1] + j*self.pool_size)
        
        #no weights to update so it just propogates the gradient matrix
        return output
    
    def backward(self, d_output, learning_rate=None):
        batch_size, channels, h, w = self.input.shape
        d_input = np.zeros_like(self.input)
        
        output_h = d_output.shape[2]
        output_w = d_output.shape[3]
        
        #set the gradiant to 0 at all positions except the max positions
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_h):
                    for j in range(output_w):
                        max_i, max_j = self.max_indices[b, c, i, j]
                        d_input[b, c, max_i, max_j] += d_output[b, c, i, j]
        
        return d_input


class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        #forward pass
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros(output_size)
        
        #backward pass
        self.input = None
        self.input_shape = None  # store original input shape for reshaping in backward pass
    
    def forward(self, input_data):
        """
        Forward pass for fully connected layer

        Args:
            input_data: shape (input_size, batch_size) or (batch_size, channels, height, width)
        Returns:
            output: shape (output_size, batch_size)
        """
        
        #flatten input to match with layer shape
        if input_data.ndim == 4:
            self.input_shape = input_data.shape
            batch_size = input_data.shape[0]
            input_data = input_data.reshape(batch_size, -1)
            input_data = input_data.T
        elif input_data.ndim == 1:
            self.input_shape = None
            input_data = input_data.reshape(-1, 1)
        elif input_data.shape[0] != self.input_size:
            self.input_shape = None
            input_data = input_data.T
        else:
            self.input_shape = None
        
        self.input = input_data
        
        output = np.dot(self.weights, input_data) + self.biases.reshape(-1, 1)
        
        return output
    
    def backward(self, d_output, learning_rate):
        batch_size = d_output.shape[1]
        
        #compute local gradients
        d_weights = np.dot(d_output, self.input.T) / batch_size
        d_biases = np.mean(d_output, axis=1)
        d_input = np.dot(self.weights.T, d_output)
        
        #update weights based on local gradients
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        
        if self.input_shape is not None:
            d_input = d_input.T.reshape(self.input_shape)
        
        return d_input


class ReLU:
    def __init__(self):
        self.input = None
    
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)
    
    def backward(self, d_output, learning_rate=None):
        d_input = d_output.copy()
        d_input[self.input <= 0] = 0
        return d_input

class Sigmoid:
    def __init__(self):
        self.output = None
    
    def forward(self, input_data):
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output
    
    def backward(self, d_output, learning_rate=None):
        d_input = d_output * self.output * (1 - self.output)
        return d_input


class CNN:
    def __init__(self, input_shape=(1, 128, 128)):

        self.layers = [ConvLayer(8, 3, input_shape[0]),
                          ReLU(),
                          MaxPoolLayer(2),
                          ConvLayer(16, 3, 8),
                          ReLU(),
                          MaxPoolLayer(2),
                          FullyConnectedLayer(14400, 64),
                          ReLU(),
                          FullyConnectedLayer(64, 1),
                          Sigmoid()]
        
    
    def forward(self, x):
        """
        Loops through layers and performs forward pass

        Args:
            x: shape (batch_size, channels, height, width)
        Returns:
            output: single sigmoid probability
        """
        current = x
        for layer in self.layers:
            current = layer.forward(current)
        return current
    
    def backward(self, d_output, learning_rate):
        """
        Loops through layers in reverse and performs backward pass
        Args:
            d_output: gradient of loss with respect to the network output
            learning_rate: learning rate for parameter updates
        Returns:
            gradient: gradient with respect to input data
        """
        gradient = d_output
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)
        return gradient
    
    def test_forward_pass(self, x):
        """
        Just runs forward pass and print shapes at each layer, not a very well written test
        """
        print("\n" + "="*50)
        print("Testing Forward Pass - Shape at each layer:")
        print("="*50)
        current = x
        print(f"Input shape: {current.shape}")
        
        for i, layer in enumerate(self.layers):
            current = layer.forward(current)
            layer_name = layer.__class__.__name__
            print(f"Layer {i+1} ({layer_name}): {current.shape if current is not None else 'None'}")
        
        print("="*50)
        return current
    
    def train_step(self, x, y, learning_rate):
        """
        Runs through one training step, which consists of:
        1. Forward pass
        2. Loss computation
        3. Backward pass
        4. Weight updates (included in backward pass functions)
        
        Args:
            x: input data (batch_size, channels, height, width)
            y: labels (batch_size, 1)
        Returns:
            loss: scalar loss value
            accuracy: scalar accuracy value
        """
        output = self.forward(x)
        
        epsilon = 1e-7 #to prevent log(0)
        loss = -np.mean(y.T * np.log(output + epsilon) + (1 - y.T) * np.log(1 - output + epsilon))
        
        predictions = (output > 0.5)
        accuracy = np.mean(predictions == y.T)

        batch_size = y.shape[0]
        d_loss = (output - y.T) / batch_size
        
        self.backward(d_loss, learning_rate)
        
        return loss, accuracy
    
    def predict(self, x):
        output = self.forward(x)
        return (output > 0.5).astype(int)
    
    def evaluate(self, x, y):
        output = self.forward(x)
        predictions = (output > 0.5).astype(int)
        accuracy = np.mean(predictions == y)
        
        epsilon = 1e-7
        output_clipped = np.clip(output, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(output_clipped) + (1 - y) * np.log(1 - output_clipped))
        
        return loss, accuracy


def load_and_preprocess_data(pickle_file='image_data.pkl', target_size=(128, 128)):
    """
    Load data from pickle file and preprocess to grayscale and given size
    """

    print(f"Loading data from {pickle_file}")
    df = pickle.load(open(pickle_file, 'rb'))
    
    print(f"Total images loaded: {len(df)}")
    
    train_data = df[df['split'] == 'train'].copy()
    val_data = df[df['split'] == 'val'].copy()
    test_data = df[df['split'] == 'test'].copy()
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    def preprocess_split(data_df):
        X_list = []
        y_list = []
        
        for idx, row in data_df.iterrows():
            img_array = row['image_array']
            
            #convert to grayscale
            if len(img_array.shape) == 3:
                img_array = np.mean(img_array, axis=2)
            
            #resize to target size
            from PIL import Image
            img = Image.fromarray(img_array.astype(np.uint8))
            img_resized = img.resize(target_size, Image.BILINEAR)
            img_array = np.array(img_resized)
            
            #normalize to [0,1]
            img_array = img_array / 255.0
            
            #add the channel dimension
            img_array = img_array.reshape(1, target_size[0], target_size[1])
            
            X_list.append(img_array)
            
            label = 1 if row['label'] == 'PNEUMONIA' else 0
            y_list.append(label)
        
        X = np.array(X_list)
        y = np.array(y_list).reshape(-1, 1)
        
        return X, y
    
    X_train, y_train = preprocess_split(train_data)
    X_val, y_val = preprocess_split(val_data)
    X_test, y_test = preprocess_split(test_data)
    
    print(f"\nPreprocessed data shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_cnn(epochs=10, batch_size=8, learning_rate=0.001):
    """
    User interface to train the CNN model
    """
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess_data()
    
    model = CNN(input_shape=(1, 128, 128))
    
    print(f"\nStarting training for {epochs} epochs")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    print("-" * 60)
    
    num_batches = len(X_train) // batch_size
    
    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        epoch_loss = 0
        epoch_accuracy = 0
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            loss, accuracy = model.train_step(X_batch, y_batch, learning_rate)
            
            epoch_loss += loss
            epoch_accuracy += accuracy
        
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        
        #test on validation set
        val_loss, val_accuracy = model.evaluate(X_val, y_val)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {avg_loss:.4f}, Acc: {avg_accuracy:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    #final test set evaluation
    print("\n" + "=" * 60)
    print("Evaluating on test set")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print("=" * 60)
    
    return model

def test_backpropagation():
    """
    This is a small example with only 10 samples to demonstrate the model's ability to learn
    """
    print("\n" + "="*60)
    print("Testing Backpropagation")
    print("="*60)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess_data()
    model = CNN(input_shape=(1, 128, 128))
    
    #select 10 random samples
    random_indices = np.random.choice(len(X_train), size=10, replace=False)
    test_batch = X_train[random_indices]
    test_labels = y_train[random_indices]
    
    print(f"\nTesting with {len(test_batch)} samples")
    print(f"True labels: {test_labels.T}")
    
    #forward pass
    print("\n" + "-"*60)
    print("BEFORE TRAINING:")
    output_before = model.forward(test_batch)
    predictions_before = (output_before > 0.5).astype(int)
    print(f"Predictions: {predictions_before}")
    print(f"Outputs: {output_before.T}")
    
    #initial loss
    epsilon = 1e-7
    loss_before = -np.mean(test_labels.T * np.log(output_before + epsilon) + 
                          (1 - test_labels.T) * np.log(1 - output_before + epsilon))
    accuracy_before = np.mean(predictions_before == test_labels.T)
    print(f"Loss: {loss_before:.4f}, Accuracy: {accuracy_before:.4f}")
    
    print("\n" + "-"*60)
    print("TRAINING (10 steps on same batch):")
    print("-"*60)
    learning_rate = 0.01
    
    for step in range(10):
        loss, acc = model.train_step(test_batch, test_labels, learning_rate)
        print(f"Epoch {step+1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
    
    print("\n" + "-"*60)
    print("AFTER TRAINING:")
    output_after = model.forward(test_batch)
    predictions_after = (output_after > 0.5).astype(int)
    print(f"Predictions: {predictions_after.T}")
    print(f"Outputs: {output_after.T}")
    
    #final loss
    loss_after = -np.mean(test_labels.T * np.log(output_after + epsilon) + 
                         (1 - test_labels.T) * np.log(1 - output_after + epsilon))
    accuracy_after = np.mean(predictions_after == test_labels.T)
    print(f"Loss: {loss_after:.4f}, Accuracy: {accuracy_after:.4f}")
    
    print("\n" + "="*60)
    print(f"Loss decreased: {loss_before:.4f} → {loss_after:.4f}")
    print(f"Accuracy improved: {accuracy_before:.4f} → {accuracy_after:.4f}")
    print("="*60)

    print("\n" + "=" * 60)
    print("Evaluating on test set (first 50 samples)")
    test_loss, test_accuracy = model.evaluate(X_test[:50], y_test[:50])
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    
    test_backpropagation()

    # Uncomment this if you are confident in your computer and want to see how it does
    #model = train_cnn(epochs=5, batch_size=4, learning_rate=0.0001)   