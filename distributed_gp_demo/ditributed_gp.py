import numpy as np
import matplotlib.pyplot as plt

colors = [
    "#5F4690",
    "#1D6996",
    "#38A6A5",
    "#0F8554",
    "#73AF48",
    "#EDAD08",
    "#E17C05",
    "#CC503E",
    "#94346E",
    "#6F4070",
    "#994E95",
    "#666666",
]


# Parameters
N = 5           # number of agents
Ti = 500         # data points per agent
M = 100          # number of inducing points (increased from 10)
d_state = 2     # per-agent state dimension (x, y)
d_act = 2       # per-agent action dimension (dx, dy)
sigma2 = 1    # noise variance
np.random.seed(69)

# Generate synthetic data
# Global state dimension = N * d_state
# Global action dimension = N * d_act
d_s = N * d_state
d_a = N * d_act
d_out = d_s

# True dynamics: simple linear with noise
W_true = np.random.randn(d_s, d_s + d_a)

# Local data per agent
X_list = []
Y_list = []
for i in range(N):
    # generate Ti samples
    s = np.random.randn(Ti, d_s)
    a = np.random.randn(Ti, d_act)
    # zero-pad actions
    a_pad = np.zeros((Ti, d_a))
    a_pad[:, i*d_act:(i+1)*d_act] = a
    X = np.hstack([s, a_pad])
    # compute next state
    Y = s + X.dot(W_true.T) + np.sqrt(sigma2) * np.random.randn(Ti, d_out)
    # delta s targets
    Y = Y - s
    X_list.append(X)
    Y_list.append(Y)

# Global data
X_all = np.vstack(X_list)
Y_all = np.vstack(Y_list)

# Sample inducing points from global data - use better coverage
# Create inducing points that cover the range better
np.random.seed(123)  # Different seed for inducing points
Z_states = np.random.uniform(X_all[:, :d_s].min(), X_all[:, :d_s].max(), (M, d_s))
Z_actions = np.random.uniform(X_all[:, d_s:].min(), X_all[:, d_s:].max(), (M, d_a))
Z = np.hstack([Z_states, Z_actions])

# Kernel function (RBF) with more appropriate lengthscale for high-dimensional data
def rbf_kernel(X1, X2, lengthscale=10.0):  # Increased further for 20D data
    sqdist = np.sum(X1**2,1).reshape(-1,1) + np.sum(X2**2,1) - 2*np.dot(X1, X2.T)
    return np.exp(-0.5/lengthscale**2 * sqdist)

# Precompute Kzz
Kzz = rbf_kernel(Z, Z)

# Compute local natural parameters J_i, h_i
J_list = []
h_list = []
for i in range(N):
    X = X_list[i]
    Y = Y_list[i]
    Kzx = rbf_kernel(Z, X)
    Kxz = Kzx.T
    Ji = (1/sigma2) * (Kzx.dot(Kxz))
    hi = (1/sigma2) * (Kzx.dot(Y))
    J_list.append(Ji)
    h_list.append(hi)

# Doubly weight stochastic communication graph
# 1. Cycle (ring) graph weights
W_cycle = np.array([
    [0.5 , 0.25, 0.  , 0.  , 0.25],
    [0.25, 0.5 , 0.25, 0.  , 0.  ],
    [0.  , 0.25, 0.5 , 0.25, 0.  ],
    [0.  , 0.  , 0.25, 0.5 , 0.25],
    [0.25, 0.  , 0.  , 0.25, 0.5 ]
])

# 2. Path graph with Metropolis weights
W_path = np.array([
    [2/3, 1/3,   0,   0,   0],
    [1/3, 1/3, 1/3,   0,   0],
    [  0, 1/3, 1/3, 1/3,   0],
    [  0,   0, 1/3, 1/3, 1/3],
    [  0,   0,   0, 1/3, 2/3]
])

# 3. Star graph with Metropolis weights
W_star = np.array([
    [1/5, 1/5, 1/5, 1/5, 1/5],
    [1/5, 4/5,   0,   0,   0],
    [1/5,   0, 4/5,   0,   0],
    [1/5,   0,   0, 4/5,   0],
    [1/5,   0,   0,   0, 4/5]
])

# Store all graphs for comparison
graphs = {
    'Cycle': W_cycle,
    'Path': W_path,
    'Star': W_star
}

# First, create convergence comparison plot
def compute_convergence_error(W_graph, max_rounds=100, num_seeds=5):
    """Compute convergence error over multiple seeds"""
    convergence_errors = []
    
    for seed in range(num_seeds):
        np.random.seed(seed)
        
        # Reinitialize data for each seed
        X_list_seed = []
        Y_list_seed = []
        for i in range(N):
            s = np.random.randn(Ti, d_s)
            a = np.random.randn(Ti, d_act)
            a_pad = np.zeros((Ti, d_a))
            a_pad[:, i*d_act:(i+1)*d_act] = a
            X = np.hstack([s, a_pad])
            Y = s + X.dot(W_true.T) + np.sqrt(sigma2) * np.random.randn(Ti, d_out)
            Y = Y - s
            X_list_seed.append(X)
            Y_list_seed.append(Y)
        
        # Compute local natural parameters for this seed
        J_list_seed = []
        h_list_seed = []
        for i in range(N):
            X = X_list_seed[i]
            Y = Y_list_seed[i]
            Kzx = rbf_kernel(Z, X)
            Kxz = Kzx.T
            Ji = (1/sigma2) * (Kzx.dot(Kxz))
            hi = (1/sigma2) * (Kzx.dot(Y))
            J_list_seed.append(Ji)
            h_list_seed.append(hi)
        
        # Compute centralized posterior for this seed
        X_all_seed = np.vstack(X_list_seed)
        Y_all_seed = np.vstack(Y_list_seed)
        KzX_all_seed = rbf_kernel(Z, X_all_seed)
        J_central_seed = (1/sigma2) * KzX_all_seed.dot(KzX_all_seed.T)
        h_central_seed = (1/sigma2) * KzX_all_seed.dot(Y_all_seed)
        S_glob_seed = np.linalg.inv(Kzz + J_central_seed)
        m_glob_seed = S_glob_seed.dot(h_central_seed)
        
        errors_this_seed = []
        
        # Track convergence over rounds
        J_cons = J_list_seed.copy()
        h_cons = h_list_seed.copy()
        
        for round_idx in range(max_rounds + 1):
            if round_idx > 0:
                J_cons = [sum(W_graph[i,j] * J_cons[j] for j in range(N)) for i in range(N)]
                h_cons = [sum(W_graph[i,j] * h_cons[j] for j in range(N)) for i in range(N)]
            
            # Compute consensus posterior
            J_bar = J_cons[0]
            h_bar = h_cons[0]
            J_sum = N * J_bar
            h_sum = N * h_bar
            
            S_cons = np.linalg.inv(Kzz + J_sum)
            m_cons = S_cons.dot(h_sum)
            
            # Compute error
            m_error = np.linalg.norm(m_cons - m_glob_seed)
            errors_this_seed.append(m_error)
        
        convergence_errors.append(errors_this_seed)
    
    return np.array(convergence_errors)

# Create convergence comparison plot
plt.style.use('seaborn-v0_8-ticks')
plt.figure(figsize=(6, 3))

# Plot 1: Convergence comparison
max_rounds = 150
rounds_range = np.arange(max_rounds + 1)

colors_graph = {'Cycle': colors[0], 'Path': colors[1], 'Star': colors[2]}

for graph_name, W_graph in graphs.items():
    print(f"Computing convergence for {graph_name} graph...")
    convergence_data = compute_convergence_error(W_graph, max_rounds=max_rounds, num_seeds=5)
    
    # Compute mean and std
    mean_error = np.maximum(np.mean(convergence_data, axis=0),1e-6)
    std_error = np.std(convergence_data, axis=0)
    
    # Plot with error bars
    plt.plot(rounds_range, mean_error, label=f'{graph_name} Graph', 
             color=colors_graph[graph_name], linewidth=2)
    plt.fill_between(rounds_range, 
                     mean_error - 1*std_error,
                     mean_error + 1*std_error,
                     color=colors_graph[graph_name], alpha=0.2)

plt.yscale('log')
plt.grid(True)
plt.xlabel('Communication Rounds')
plt.ylabel(r'Convergence Error')
plt.legend()
# Use path graph for subsequent analysis
W = W_path
plt.tight_layout()
plt.savefig('convergence_comparison.pdf', bbox_inches='tight')

# Now create the original subplot comparison for different rounds
plt.figure(figsize=(12, 10))

rounds = [0, 5, 10, 50]  # Rounds to visualize

for k in range(4):
    np.random.seed(42)

    K_rounds = rounds[k]

    # Initialize consensus variables
    J_cons = J_list.copy()
    h_cons = h_list.copy()

    for _ in range(K_rounds):
        J_cons = [sum(W[i,j] * J_cons[j] for j in range(N)) for i in range(N)]
        h_cons = [sum(W[i,j] * h_cons[j] for j in range(N)) for i in range(N)]

    # After consensus, average (all should be equal)
    J_bar = J_cons[0]
    h_bar = h_cons[0]
    # Recover global sum
    J_sum = N * J_bar
    h_sum = N * h_bar

    # Compute consensus posterior
    S_cons = np.linalg.inv(Kzz + J_sum)
    m_cons = S_cons.dot(h_sum)

    # Compute centralized posterior directly
    KzX_all = rbf_kernel(Z, X_all)
    J_central = (1/sigma2) * KzX_all.dot(KzX_all.T)
    h_central = (1/sigma2) * KzX_all.dot(Y_all)
    S_glob = np.linalg.inv(Kzz + J_central)
    m_glob = S_glob.dot(h_central)

    # Compare
    m_diff = np.linalg.norm(m_cons - m_glob)
    S_diff = np.linalg.norm(S_cons - S_glob)

    print({
        'Metric': ['||m_cons - m_glob||', '||S_cons - S_glob||'],
        'Value': [m_diff, S_diff]
    })

    # Generate synthetic test data for a varying first state and fixed actions
    test_s = np.linspace(-2, 2, 5).reshape(-1, 1)  # Wider range, closer to training
    test_s2 = np.linspace(-4, 3, 5).reshape(-1, 1)  # Wider range, closer to training
    test_s = np.hstack([test_s, test_s2, np.zeros((5, d_s - 2))])  # pad to full state dimension d_s
    # Use actions closer to training distribution
    test_a = np.zeros((5, d_a))  # Start with zeros like training
    test_a[:, :2] = np.random.uniform(-.5, .5, (5, 2))  # Set first agent actions to a small range
    test_X = np.hstack([test_s, test_a])

    # Compute true predictions (deterministic, no noise)
    pred_true = test_X.dot(W_true.T)  # This gives the true delta (change in state)

    # Compute predictions using consensus posterior
    K_test = rbf_kernel(Z, test_X).T
    pred_cons = K_test.dot(m_cons)
    # Compute predictions using global posterior
    pred_glob = K_test.dot(m_glob)

    # Plot predictions
    plt.subplot(2, 2, k+1)
    # True deterministic prediction
    plt.plot(test_s[:,0], pred_cons[:,0], label='Consensus Prediction', color=colors[0])
    plt.scatter(test_s[:,0], pred_true[:,0], label='True Data', color=colors[1])
    plt.plot(test_s[:,0], pred_glob[:,0], label='Global Prediction', color=colors[2])
    plt.fill_between(test_s[:,0], 
                    pred_cons[:,0] - 2* np.sqrt(np.diag(K_test.dot(S_cons).dot(K_test.T))),
                    pred_cons[:,0] + 2* np.sqrt(np.diag(K_test.dot(S_cons).dot(K_test.T))),
                    color=colors[0], alpha=0.2)
    plt.fill_between(test_s[:,0], 
                    pred_glob[:,0] - 2* np.sqrt(np.diag(K_test.dot(S_glob).dot(K_test.T))),
                    pred_glob[:,0] + 2* np.sqrt(np.diag(K_test.dot(S_glob).dot(K_test.T))),
                    color=colors[2], alpha=0.2)
    plt.title(f'Communication Rounds: {K_rounds}')
    plt.grid(True)
    plt.xlabel(r'$x_t$')
    plt.ylabel(r'$x_{t+1}$')
    if k == 0:
        plt.legend()
# Reduce vertical spacing between subplots
plt.tight_layout(rect=[0, 0, 1, 0.5], h_pad=1.0)
plt.savefig('predictions_comparison.pdf', bbox_inches='tight')


