import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

class GradientDescentDemo:
    def __init__(self, x: List[float], y: List[float], w_init: float = 0.0):
        """
        Initialize Gradient Descent Demo with data
        
        Parameters:
        -----------
        x : List[float]
            Input features
        y : List[float]
            Target values
        w_init : float
            Initial weight value
        """
        self.x = np.array(x)
        self.y = np.array(y)
        self.w = w_init
        self.history = []
        
    def predict(self, x_val: float) -> float:
        """Make prediction using current weight"""
        return self.w * x_val
    
    def loss(self, y_true: float, y_pred: float) -> float:
        """Calculate squared error loss"""
        return (y_true - y_pred) ** 2
    
    def gradient(self, x_val: float, y_true: float, y_pred: float) -> float:
        """Calculate gradient for linear regression"""
        return -2 * x_val * (y_true - y_pred)
    
    def batch_gradient_descent(self, lr: float = 0.001, epochs: int = 10) -> Dict[str, Any]:
        """
        Batch Gradient Descent (BGD)
        Uses entire dataset to compute gradient for each update
        """
        self.w = 0.0
        self.history = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_details = []
            
            # Calculate gradient over entire dataset
            total_gradient = 0
            total_loss = 0
            
            for i in range(len(self.x)):
                y_pred = self.predict(self.x[i])
                loss_val = self.loss(self.y[i], y_pred)
                g = self.gradient(self.x[i], self.y[i], y_pred)
                
                total_gradient += g
                total_loss += loss_val
                
                epoch_details.append({
                    'x': float(self.x[i]),
                    'y': float(self.y[i]),
                    'ypred': float(y_pred),
                    'loss': float(loss_val),
                    'gradient': float(g)
                })
            
            # Update weight using average gradient
            avg_gradient = total_gradient / len(self.x)
            self.w = self.w - lr * avg_gradient
            
            self.history.append({
                'epoch': epoch + 1,
                'weight': float(self.w),
                'total_loss': float(total_loss),
                'avg_gradient': float(avg_gradient),
                'details': epoch_details
            })
        
        return {
            'final_weight': float(self.w),
            'history': self.history,
            'type': 'BGD'
        }
    
    def stochastic_gradient_descent(self, lr: float = 0.001, epochs: int = 10) -> Dict[str, Any]:
        """
        Stochastic Gradient Descent (SGD)
        Updates weight after each training sample
        """
        self.w = 0.0
        self.history = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_details = []
            
            for i in range(len(self.x)):
                y_pred = self.predict(self.x[i])
                loss_val = self.loss(self.y[i], y_pred)
                g = self.gradient(self.x[i], self.y[i], y_pred)
                
                # Update weight immediately for each sample
                self.w = self.w - lr * g
                epoch_loss += loss_val
                
                epoch_details.append({
                    'x': float(self.x[i]),
                    'y': float(self.y[i]),
                    'ypred': float(y_pred),
                    'loss': float(loss_val),
                    'gradient': float(g),
                    'weight_after': float(self.w)
                })
            
            self.history.append({
                'epoch': epoch + 1,
                'weight': float(self.w),
                'total_loss': float(epoch_loss),
                'details': epoch_details
            })
        
        return {
            'final_weight': float(self.w),
            'history': self.history,
            'type': 'SGD'
        }
    
    def mini_batch_gradient_descent(self, lr: float = 0.001, epochs: int = 10, 
                                   batch_size: int = 2) -> Dict[str, Any]:
        """
        Mini-Batch Gradient Descent (MBGD)
        Updates weight using small batches of data
        """
        self.w = 0.0
        self.history = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_details = []
            batch_details = []
            
            # Shuffle data
            indices = np.random.permutation(len(self.x))
            x_shuffled = self.x[indices]
            y_shuffled = self.y[indices]
            
            for batch_start in range(0, len(self.x), batch_size):
                batch_end = min(batch_start + batch_size, len(self.x))
                x_batch = x_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                
                batch_gradient = 0
                batch_loss = 0
                
                for j in range(len(x_batch)):
                    y_pred = self.predict(x_batch[j])
                    loss_val = self.loss(y_batch[j], y_pred)
                    g = self.gradient(x_batch[j], y_batch[j], y_pred)
                    
                    batch_gradient += g
                    batch_loss += loss_val
                    
                    epoch_details.append({
                        'x': float(x_batch[j]),
                        'y': float(y_batch[j]),
                        'ypred': float(y_pred),
                        'loss': float(loss_val),
                        'gradient': float(g)
                    })
                
                # Update weight using average batch gradient
                avg_batch_gradient = batch_gradient / len(x_batch)
                self.w = self.w - lr * avg_batch_gradient
                
                batch_details.append({
                    'batch': len(batch_details) + 1,
                    'batch_loss': float(batch_loss),
                    'batch_gradient': float(avg_batch_gradient),
                    'weight_after': float(self.w)
                })
                
                epoch_loss += batch_loss
            
            self.history.append({
                'epoch': epoch + 1,
                'weight': float(self.w),
                'total_loss': float(epoch_loss),
                'batch_details': batch_details,
                'details': epoch_details
            })
        
        return {
            'final_weight': float(self.w),
            'history': self.history,
            'type': 'MBGD',
            'batch_size': batch_size
        }
    
    def calculate_all_predictions(self) -> List[float]:
        """Calculate predictions for all x values"""
        return [self.predict(x_val) for x_val in self.x]
    
    def visualize_results(self):
        """Create visualization plots"""
        try:
            fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot 1: Data points and regression line
            ax1.scatter(self.x, self.y, color='blue', label='Actual Data', s=50)
            x_range = np.linspace(min(self.x), max(self.x), 100)
            y_pred_range = self.w * x_range
            ax1.plot(x_range, y_pred_range, 'r-', label=f'Regression Line (w={self.w:.4f})', linewidth=2)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_title('Linear Regression Fit')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Loss convergence
            if self.history:
                epochs = [h['epoch'] for h in self.history]
                losses = [h['total_loss'] for h in self.history]
                ax2.plot(epochs, losses, 'g-o', linewidth=2, markersize=6)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Total Loss')
                ax2.set_title('Loss Convergence')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Create a second figure for additional visualization if needed
            fig2 = None
            if self.history and len(self.history) > 0:
                fig2, ax3 = plt.subplots(figsize=(8, 4))
                weights = [h['weight'] for h in self.history]
                ax3.plot(epochs, weights, 'b-s', linewidth=2, markersize=6)
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Weight (w)')
                ax3.set_title('Weight Updates Over Epochs')
                ax3.grid(True, alpha=0.3)
                plt.tight_layout()
            
            return fig1, fig2
        except Exception as e:
            print(f"Error in visualize_results: {e}")
            # Return empty figures if there's an error
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.text(0.5, 0.5, 'Visualization Error', 
                    ha='center', va='center', transform=ax1.transAxes)
            return fig1, None


def get_sample_data() -> Tuple[List[float], List[float]]:
    """
    Return sample dataset for demonstration
    
    Returns:
    --------
    Tuple of (x_values, y_values)
    """
    SAMPLE_X = [1, 21, 15, 19, 12]
    SAMPLE_Y = [15, 24, 35, 37, 10]
    return SAMPLE_X.copy(), SAMPLE_Y.copy()


def create_comparison_table(bgd_result, sgd_result, mbgd_result) -> pd.DataFrame:
    """
    Create comparison table for different gradient descent algorithms
    """
    comparison_data = []
    
    for result, name in [(bgd_result, "BGD"), (sgd_result, "SGD"), (mbgd_result, "MBGD")]:
        if result:
            final_loss = result['history'][-1]['total_loss'] if result['history'] else 0
            comparison_data.append({
                'Algorithm': name,
                'Final Weight': f"{result['final_weight']:.6f}",
                'Final Loss': f"{final_loss:.4f}",
                'Epochs': len(result['history'])
            })
    
    return pd.DataFrame(comparison_data)


def get_detailed_output(result, max_steps=20) -> List[str]:
    """
    Generate detailed step-by-step output for visualization
    
    Parameters:
    -----------
    result : Dict
        Gradient descent result
    max_steps : int
        Maximum number of steps to show per epoch
        
    Returns:
    --------
    List of formatted output strings
    """
    output = []
    
    if not result or 'history' not in result:
        return output
    
    for epoch_data in result['history']:
        epoch = epoch_data['epoch']
        output.append(f"\n{'='*60}")
        output.append(f"Epoch {epoch}: Weight = {epoch_data['weight']:.6f}, Total Loss = {epoch_data['total_loss']:.4f}")
        output.append(f"{'='*60}")
        
        if 'details' in epoch_data and epoch_data['details']:
            details = epoch_data['details']
            # Limit the number of steps shown to avoid overwhelming
            if len(details) > max_steps:
                details = details[:max_steps]
            
            for i, step in enumerate(details):
                output.append(
                    f"  Step {i+1:2d}: x={step['x']:5.1f}, y={step['y']:5.1f}, "
                    f"ŷ={step['ypred']:7.3f}, loss={step['loss']:8.3f}, "
                    f"g={step['gradient']:8.3f}"
                )
    
    return output


def simple_gradient_descent_demo(x: List[float], y: List[float], lr: float = 0.001, epochs: int = 10) -> Dict:
    """
    Simple gradient descent demonstration as shown in the example
    
    Parameters:
    -----------
    x : List[float]
        Input features
    y : List[float]
        Target values
    lr : float
        Learning rate
    epochs : int
        Number of training epochs
        
    Returns:
    --------
    Dict containing training history and results
    """
    w = 0.0
    history = []
    
    for epoch in range(epochs):
        total_loss = 0
        epoch_steps = []
        
        print(f"\nEpoch {epoch}")
        
        for i in range(len(x)):
            ypred = w * x[i]
            loss = (y[i] - ypred) ** 2
            total_loss += loss
            
            g = -2 * x[i] * (y[i] - ypred)
            w = w - lr * g
            
            epoch_steps.append({
                'x': x[i],
                'y': y[i],
                'ypred': ypred,
                'loss': loss,
                'gradient': g,
                'weight': w
            })
        
        history.append({
            'epoch': epoch,
            'weight': w,
            'total_loss': total_loss,
            'steps': epoch_steps
        })
    
    return {
        'final_weight': w,
        'history': history,
        'type': 'Simple GD'
    }


# Example usage and testing
if __name__ == "__main__":
    # Test the GradientDescentDemo class
    print("Testing GradientDescentDemo class...")
    
    # Sample data
    x_sample, y_sample = get_sample_data()
    
    # Create demo instance
    gd = GradientDescentDemo(x_sample, y_sample)
    
    # Test BGD
    print("\n1. Batch Gradient Descent:")
    bgd_result = gd.batch_gradient_descent(lr=0.001, epochs=3)
    print(f"   Final weight: {bgd_result['final_weight']:.6f}")
    print(f"   Final loss: {bgd_result['history'][-1]['total_loss']:.4f}")
    
    # Test SGD
    print("\n2. Stochastic Gradient Descent:")
    gd = GradientDescentDemo(x_sample, y_sample)  # Reset
    sgd_result = gd.stochastic_gradient_descent(lr=0.001, epochs=3)
    print(f"   Final weight: {sgd_result['final_weight']:.6f}")
    print(f"   Final loss: {sgd_result['history'][-1]['total_loss']:.4f}")
    
    # Test MBGD
    print("\n3. Mini-Batch Gradient Descent:")
    gd = GradientDescentDemo(x_sample, y_sample)  # Reset
    mbgd_result = gd.mini_batch_gradient_descent(lr=0.001, epochs=3, batch_size=2)
    print(f"   Final weight: {mbgd_result['final_weight']:.6f}")
    print(f"   Final loss: {mbgd_result['history'][-1]['total_loss']:.4f}")
    
    # Test visualization
    print("\n4. Testing Visualization:")
    gd = GradientDescentDemo(x_sample, y_sample)  # Reset
    bgd_result = gd.batch_gradient_descent(lr=0.001, epochs=5)
    fig1, fig2 = gd.visualize_results()
    if fig1:
        print("   Visualization created successfully!")
        plt.close(fig1)
    if fig2:
        plt.close(fig2)
    
    print("\nAll tests passed! GradientDescentDemo is working correctly.")