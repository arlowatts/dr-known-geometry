import os
import time
import argparse
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from testing.parameter_extraction.parameter_optimization_V2 import optimize

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run parameter optimization')
    parser.add_argument('--method', type=str, default='cuda_ad_rgb', help='Optimization method')
    parser.add_argument('--geometry', type=str, default='./geometry/bunny/bunny.obj', help='Path to geometry file')
    parser.add_argument('--iterations', type=int, default=200, help='Number of iterations')
    
    args = parser.parse_args()
    
    # Create output directory for any files we might need to save
    output_dir = os.path.join(os.path.dirname(__file__), "optimization_results")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("Running optimization...")
        print(f"Method: {args.method}, Geometry: {args.geometry}, Iterations: {args.iterations}")
        
        result = optimize(args.method, args.geometry, args.iterations)
        
        print("Optimization completed successfully")
        return True
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    main()
