"""
CNN Framework for NORMAL vs PNEUMONIA Classification

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
        return np.zeros_like(self.input)


class MaxPoolLayer:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size
        self.input = None
        self.max_indices = None
    
    def forward(self, input_data):
        """
        Forward pass for max pooling
        
        TODO STEP 2: Implement max pooling
        - Divide input into pool_size x pool_size regions
        - Take maximum value from each region
        - Store the position of max value (needed for backprop)
        
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
        
        return output
    
    def backward(self, d_output, learning_rate=None):
        return np.zeros_like(self.input)


class FullyConnectedLayer:
    """Fully connected (dense) layer implementation"""
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        #forward pass
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros(output_size)
        
        #backward pass
        self.input = None
    
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
            batch_size = input_data.shape[0]
            input_data = input_data.reshape(batch_size, -1)
            input_data = input_data.T  # (flattened_features, batch_size)
        elif input_data.ndim == 1:
            input_data = input_data.reshape(-1, 1)
        elif input_data.shape[0] != self.input_size:
            input_data = input_data.T
        
        self.input = input_data
        
        output = np.dot(self.weights, input_data) + self.biases.reshape(-1, 1)
        
        return output
    
    def backward(self, d_output, learning_rate):
        if self.input is not None:
            if self.input.ndim == 1:
                return np.zeros_like(self.input)
            return np.zeros((self.input_size, self.input.shape[1]))
        return np.zeros((self.input_size, 1))


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
        return d_output


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
            d_output: gradient of loss with respect to itself
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
        Just runs forward pass and print shapes at each layer
        """
        print("\n" + "="*60)
        print("Testing Forward Pass - Shape at each layer:")
        print("="*60)
        current = x
        print(f"Input shape: {current.shape}")
        
        for i, layer in enumerate(self.layers):
            current = layer.forward(current)
            layer_name = layer.__class__.__name__
            print(f"Layer {i+1} ({layer_name}): {current.shape if current is not None else 'None'}")
        
        print("="*60)
        return current
    
    def train_step(self, x, y, learning_rate):
        """
        STEP 3 TODO: For now, just implement forward pass and loss computation
        Don't worry about backward pass yet!
        
        Args:
            x: input data (batch_size, channels, height, width)
            y: labels (batch_size, 1)
        Returns:
            loss: scalar loss value
            accuracy: scalar accuracy value
        """
        output = self.forward(x)
        
        # loss = -mean(y * log(output) + (1-y) * log(1-output))
        # Use epsilon to avoid log(0)
        epsilon = 1e-7
        loss = -np.mean(y * np.log(output + epsilon) + (1 - y) * np.log(1 - output + epsilon))
        
        # predictions = (output > 0.5)
        # accuracy = mean(predictions == y)
        accuracy = None
        
        #self.backward(d_loss, learning_rate)
        
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
    
    # Prepare data by split
    train_data = df[df['split'] == 'train'].copy()
    val_data = df[df['split'] == 'val'].copy()
    test_data = df[df['split'] == 'test'].copy()
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    def preprocess_split(data_df):
        X_list = []
        y_list = []
        
        for idx, row in data_df.iterrows():
            img_array = row['image_array']
            
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                img_array = np.mean(img_array, axis=2)
            
            # Resize image
            from PIL import Image
            img = Image.fromarray(img_array.astype(np.uint8))
            img_resized = img.resize(target_size, Image.BILINEAR)
            img_array = np.array(img_resized)
            
            # Normalize to [0, 1]
            img_array = img_array / 255.0
            
            # Add channel dimension
            img_array = img_array.reshape(1, target_size[0], target_size[1])
            
            X_list.append(img_array)
            
            # Convert label to binary (0 for NORMAL, 1 for PNEUMONIA)
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
    Train the CNN model
    """
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess_data()
    
    # Initialize model
    model = CNN(input_shape=(1, 128, 128))
    
    # Training loop
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
        val_loss, val_accuracy = model.evaluate(X_val[:32], y_val[:32])
        
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


if __name__ == "__main__":
    
    print("\nTesting with a single sample")
    (X_train, y_train), _, _ = load_and_preprocess_data()
    model = CNN(input_shape=(1, 128, 128))
    output = model.test_forward_pass(X_train[0:1])
    print(f"True label: {y_train[0]}")
    print(f"\nFinal output: {output}") 
    if output > 0.5:
        print("Predicted: PNEUMONIA")
    else:
        print("Predicted: NORMAL")   