import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def f(x):
    """Function: f(x) = 0.5*e^x - 5x + 2"""
    return 0.5 * np.exp(x) - 5 * x + 2

def bisection_method(a, b, tolerance=0.00001, max_iterations=100):
    """
    Bisection Method to find root of f(x)

    Parameters:
    a, b: initial guesses (must bracket the root)
    tolerance: stopping criteria
    max_iterations: maximum number of iterations

    Returns:
    results: DataFrame with all iterations
    root: final root value
    """

    # Check if initial guesses bracket a root
    if f(a) * f(b) >= 0:
        print("Error: f(a) and f(b) must have opposite signs!")
        return None, None

    # Store results
    results = []

    for i in range(max_iterations):
        c = (a + b) / 2  # midpoint
        fa = f(a)
        fb = f(b)
        fc = f(c)
        error = abs(b - a)

        # Store iteration data
        results.append({
            'iteration': i + 1,
            'guess_a': a,
            'guess_b': b,
            'midpoint_c': c,
            'f(a)': fa,
            'f(b)': fb,
            'f(c)': fc,
            'error': error
        })

        # Check for convergence
        if error < tolerance:
            print(f"Converged in {i + 1} iterations!")
            break

        # Update interval
        if fa * fc < 0:
            b = c
        else:
            a = c

    df = pd.DataFrame(results)
    return df, c

def plot_function_and_root(root, a_init, b_init):
    """Plot the function and show the root"""
    x = np.linspace(a_init - 1, b_init + 1, 1000)
    y = f(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='f(x) = 0.5e^x - 5x + 2')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.plot(root, f(root), 'ro', markersize=10, label=f'Root ≈ {root:.6f}')
    plt.grid(True, alpha=0.3)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title('Bisection Method - Root Finding', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.show()

def plot_convergence(df):
    """Plot the convergence of the error"""
    plt.figure(figsize=(10, 5))
    plt.semilogy(df['iteration'], df['error'], 'b-o', linewidth=2, markersize=6)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Error (log scale)', fontsize=12)
    plt.title('Convergence of Bisection Method', fontsize=14, fontweight='bold')
    plt.show()

# ============================================
# MAIN EXECUTION
# ============================================

print("="*60)
print("BISECTION METHOD - ROOT FINDING")
print("Function: f(x) = 0.5*e^x - 5x + 2")
print("="*60)

# Find Root 1 (between 0 and 1)
print("\n🔍 Finding Root 1 (initial guess: [0, 1])...")
df1, root1 = bisection_method(0, 1, tolerance=0.00001)

if df1 is not None:
    print(f"\n✅ Root 1 = {root1:.10f}")
    print(f"✅ f(root1) = {f(root1):.2e}")
    print("\nIteration Table (first 10 rows):")
    print(df1.head(10).to_string(index=False))

    # Plot
    plot_function_and_root(root1, 0, 1)
    plot_convergence(df1)

# Find Root 2 (between 3 and 4)
print("\n" + "="*60)
print("\n🔍 Finding Root 2 (initial guess: [3, 4])...")
df2, root2 = bisection_method(3, 4, tolerance=0.00001)

if df2 is not None:
    print(f"\n✅ Root 2 = {root2:.10f}")
    print(f"✅ f(root2) = {f(root2):.2e}")
    print("\nIteration Table (first 10 rows):")
    print(df2.head(10).to_string(index=False))

    # Plot
    plot_function_and_root(root2, 3, 4)
    plot_convergence(df2)

# Summary
print("\n" + "="*60)
print("📊 SUMMARY")
print("="*60)
print(f"Root 1: {root1:.10f}")
print(f"Root 2: {root2:.10f}")
print(f"Tolerance: 0.00001")
print("="*60)
