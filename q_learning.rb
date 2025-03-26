require "csv"
require "torch-rb"
require_relative "common/global"
require_relative "common/utility"

# Q-Learning algorithm implementation
# This class implements the Q-learning algorithm, an off-policy TD control method
# that learns the optimal policy regardless of what policy it follows
class QLearning
  ALPHA = 0.1      # Learning rate
  GAMMA = 0.9      # Discount factor
  EPSILON = 0.2    # Exploration probability
  DELTA = 1e-10    # Small constant to avoid division by zero

  attr_reader :state_transitions, :rewards, :start_state, :terminal_state
  attr_accessor :q_values, :score_log

  def initialize(state_transitions:, rewards:)
    @state_transitions = state_transitions
    @rewards = rewards
    @start_state = 12
    @terminal_state = 3
    # Initialize Q-table with zeros - state-action pairs
    @q_values = Torch.zeros(16, 4)  # 16 states, 4 possible actions
    @score_log = []
  end

  # Q-learning update rule:
  # Q(s,a) = Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
  def q_learning_update(state, action, reward, next_state)
    # Calculate max Q-value for next state
    max_next_q = next_state == terminal_state ? 0 : q_values[next_state].max.item
    # TD target
    td_target = reward + (GAMMA * max_next_q)
    # TD error
    td_error = td_target - q_values[state, action]
    # Update Q-value
    q_values[state, action] = q_values[state, action] + (ALPHA * td_error)
  end

  # Select action using epsilon-greedy policy
  def select_action(state)
    if rand < EPSILON
      # Exploration: random action
      rand(4)
    else
      # Exploitation: greedy action
      Torch.argmax(q_values[state]).to_i
    end
  end

  # Test agent using current Q-values
  def test_agent
    state = start_state
    steps = 0
    total_rewards = 0
    states_log = []
    
    while (state != terminal_state) && (steps < 30)
      states_log.append(state)
      # Always choose greedy action during testing
      action = Torch.argmax(q_values[state]).to_i
      state = transit_state(state, action: action)
      total_rewards += rewards[state]
      steps += 1
    end
    
    states_log.append(state)
    [total_rewards, states_log]
  end

  # Main training loop
  def perform
    1000.times do |episode|
      state = start_state
      steps = 0

      # Run episode
      loop do
        # Select action using epsilon-greedy policy
        action = select_action(state)
        
        # Take action and observe next state and reward
        next_state = transit_state(state, action: action)
        reward = rewards[next_state]
        
        # Update Q-values
        q_learning_update(state, action, reward, next_state)
        
        # Move to next state
        state = next_state
        steps += 1
        
        # End episode if terminal state reached or max steps
        break if (state == terminal_state) || (steps >= 30)
      end

      # Record performance metrics for this episode
      score_log.append(test_agent[0])
      
      # Print progress every 100 episodes
      if (episode + 1) % 100 == 0
        puts "Episode #{episode + 1}/1000 completed"
      end
    end
  end

  # Visualize the learned policy
  def visualize_policy
    policy = Torch.zeros(16)
    16.times do |state|
      policy[state] = Torch.argmax(q_values[state]).item if q_values[state].sum.item != 0
    end
    
    # Map actions to directions for clearer visualization
    action_to_str = {
      0 => "→",  # Right
      1 => "↓",  # Down
      2 => "←",  # Left
      3 => "↑"   # Up
    }
    
    policy_str = policy.to_a.map { |a| action_to_str[a] || "·" }
    policy_str[terminal_state] = "G"  # Mark goal state
    
    Utility.table_plot(Torch.tensor(policy_str).reshape(4, 4), title: "Learned Policy", padding: [0, 1])
  end

  private

    def transit_state(state, action: rand(4))
      state_transitions.dig(state, action)
    end
end

# Main program
state_transitions = CSV.read(File.expand_path("state_transitions.csv", __dir__), converters: :numeric)
Utility.table_print(state_transitions, title: "state_transitions data table")

# Set up the rewards: Goal state has +10 reward, some states have negative rewards (obstacles)
rewards = Torch.zeros(16)
rewards[3] = 10    # Goal state with positive reward
rewards[2] = -1    # Obstacle states with negative rewards
rewards[11] = -1
rewards[10] = -1
Utility.table_plot(rewards.type(:int).reshape(4, 4), title: "rewards", padding: [0, 1])

# Create and train Q-learning agent
puts "Training Q-learning agent..."
ql = QLearning.new(state_transitions: state_transitions, rewards: rewards)
ql.perform

# Visualize Q-values
puts "Q-values after training:"
q_max = ql.q_values.max(dim: 1)[0]
Utility.table_plot(q_max.reshape(4, 4), title: "max Q-values", padding: [0, 1])

# Plot learning curve
puts "Learning curve:"
Utility.pyplot(Array(0..ql.score_log.length.pred), ql.score_log.map(&:to_f), title: "Q-learning performance")

# Visualize learned policy
ql.visualize_policy

# Test and visualize agent trajectory
puts "Testing agent with learned policy..."
_, state_log = ql.test_agent
state_view = Torch.zeros(16)
state_view[state_log] = 1
Utility.block_plot(state_view.type(:int).reshape(4, 4), title: "Agent Trajectory")

puts "Q-learning completed successfully!"