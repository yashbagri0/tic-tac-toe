import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt


REWARDS = {
    "draw": 0.5,
    "lose": -1,
    "win": 1,
    "step": 0
}
# rewards, the more the reward for losses, the more traumatized the bot

GARBAGE_VALUE = np.inf

class TicTacToe:
    def __init__(self, size: tuple):
        self.size = size
        self.reset()

    def reset(self):
        """Reset the board to start a new game."""
        self.board = np.zeros(self.size, dtype=np.uint16)
        self.playable_board = np.full(self.size, '-', dtype=str)
        self.done = False
        return self.board

    def show_board(self):
        """Print the board."""
        print(self.board)
    
    def show_board_playable(self):
        for row in self.playable_board:
            print(' '.join(row))

    def check_winner(self):
        """Check if a player has won or if the game is a draw."""
        for i in range(self.size[0]):
            # Check rows and columns
            if np.all(self.board[i, :] == 1) or np.all(self.board[:, i] == 1):
                return 1  # 'x' wins
            if np.all(self.board[i, :] == 2) or np.all(self.board[:, i] == 2):
                return 2  # 'o' wins

        # Check diagonals
        if np.all(np.diag(self.board) == 1) or np.all(np.diag(np.fliplr(self.board)) == 1):
            return 1  # 'x' wins
        if np.all(np.diag(self.board) == 2) or np.all(np.diag(np.fliplr(self.board)) == 2):
            return 2  # 'o' wins

        # Check for a draw (if no empty spaces left)
        if not np.any(self.board == 0):
            return 0  # Draw

        return None  # Game still ongoing

    def turn(self, position, mark, playing=False):
        """Make a move and return the new state, reward."""
        x, y = position

        # Make the move
        self.board[x, y] = 1 if mark == 'x' else 2

        if playing: self.playable_board[x, y] = mark 

        # Check game state after move
        winner = self.check_winner()
        
        if winner == 1:  # 'x' wins
            self.done = True
            if playing:
                return self.board, GARBAGE_VALUE, winner
            else:
                if mark == 'x':
                    return self.board, REWARDS["win"]
                else:
                    return self.board, REWARDS["lose"]
        elif winner == 2:  # 'o' wins
            self.done = True
            if playing:
                return self.board, GARBAGE_VALUE, winner
            else:
                if mark == 'o':
                    return self.board, REWARDS["win"]
                else:
                    return self.board, REWARDS["lose"]
        elif winner == 0:  # Draw
            self.done = True
            if playing:
                return self.board, GARBAGE_VALUE, winner
            else:
                return self.board, REWARDS["draw"]

        # Game is still ongoing
        if playing:
            return self.board, GARBAGE_VALUE, GARBAGE_VALUE
        else:
            return self.board, REWARDS["step"]


class QLearningAgent:
    def __init__(self, game_board, epsilon=0.1, discount=0.1, lr=0.1, q_table=None):
        self.game_board = game_board
        self.game_size = self.game_board.size
        self.total_moves = [tuple(map(int, move)) for move in np.argwhere(self.game_board.board == 0)]
        self.epsilon = epsilon
        self.discount = discount
        self.lr = lr

        self.stats = []
        self.episode_rewards = []
        self.wins = []
        self.losses = []
        self.draws = []

        if q_table is not None:
            self.q_table = q_table
        else:
            self.q_table = {}
    
    def make_state_key(self, state):
        return str(state.flatten())[1:-1]
    
    def get_q_values(self, state):
        """Get Q-values for a state, initializing if not present"""
        state_key = self.make_state_key(state)
        if state_key not in self.q_table:
            # Initialize Q-values for all possible moves in this state, optimistic start of 0.2
            self.q_table[state_key] = np.ones(self.game_size[0] * self.game_size[1]) * 0.2
        return self.q_table[state_key]

    def update_q_value(self, state, action, new_value):
        """Update Q-value for a state-action pair"""
        state_key = self.make_state_key(state)
        action_idx = self.total_moves.index(action)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.ones(self.game_size[0] * self.game_size[1]) * 0.1
        self.q_table[state_key][action_idx] = new_value

    def get_valid_move(self, state):
        return [tuple(map(int, move)) for move in np.argwhere(state == 0)]

    def get_valid_move_index(self, valid_move):
        return [self.total_moves.index(item) for item in valid_move]

    def choose_action(self, state, playing=False):
        """Choose action using epsilon-greedy policy"""
        valid_moves = self.get_valid_move(state)
        if not playing and np.random.random() < self.epsilon:
            return valid_moves[np.random.randint(len(valid_moves))]
            
        q_values = self.get_q_values(state)
        valid_move_indices = self.get_valid_move_index(valid_moves)
        valid_q_values = q_values[valid_move_indices]
        best_indices = np.where(valid_q_values == np.max(valid_q_values))[0]  # Get all indices with the max Q-value
        best_valid_move_idx = valid_move_indices[np.random.choice(best_indices)]  # Pick one randomly
        return self.total_moves[best_valid_move_idx]

    def update_q_table_reverse(self, states, actions, final_reward):
        """Update Q-values for all state-action pairs in episode"""
        opponent_won = len(states) > len(actions)
        for i in range(len(actions) - 1, -1, -1):
            state = states[i]
            action = actions[i]
            action_idx = self.total_moves.index(action)
            current_q = self.get_q_values(state)[action_idx]
            
            if i == len(actions) - 1:  # Last move
                if opponent_won:
                    new_q = current_q + self.lr * (REWARDS['lose'] - current_q)
                else:
                    # Game ended with our move (we won or drew)
                    new_q = current_q + self.lr * (final_reward - current_q)
            else:
                next_state = states[i + 1]
                next_valid_moves = self.get_valid_move_index(self.get_valid_move(next_state))
                if not opponent_won and len(next_valid_moves) > 0:
                    next_q_values = self.get_q_values(next_state)[next_valid_moves]
                    future_value = np.max(next_q_values)
                    new_q = current_q + self.lr * (REWARDS['step'] + self.discount * future_value - current_q)
                else:
                    new_q = current_q + self.lr * (REWARDS['step'] - current_q)
        
            self.update_q_value(state, action, new_q)

    def forward(self, episodes=100000):
        
        for episode in tqdm(range(episodes), desc="Training Progress"):
            mark_x_or_o = ['x', 'o']
            mark_for_agent = np.random.choice(mark_x_or_o)
            mark_x_or_o.remove(mark_for_agent)
            mark_for_opponent = mark_x_or_o[0]
            turn = 0 if mark_for_agent == 'x' else 1
            self.game_board.reset()

            states = []
            actions = []
            episode_reward = 0
            
            while not self.game_board.done:
                if turn % 2 == 0:
                    state = self.game_board.board.copy()
                    action = self.choose_action(state)
                    # self.update_q_table(state, action, reward, next_state)

                    states.append(state) #state after opponent move
                    actions.append(action) #action we took

                    next_state, reward = self.game_board.turn(action, mark=mark_for_agent)
                    episode_reward += reward
                    
                else:
                    state = self.game_board.board
                    valid_moves = self.get_valid_move(state)
                    action = valid_moves[np.random.randint(len(valid_moves))]
                    next_state, _ = self.game_board.turn(action, mark=mark_for_opponent)
                    if self.game_board.done:
                        reward = REWARDS['lose']
                        states.append(next_state.copy())
                
                turn += 1

            self.episode_rewards.append(episode_reward)
            # I'M PRETTY SURE THE GRAPH IS WRONG BUT DIDN'T DELVED IT IN CAUSE N TIME
            if reward == REWARDS["win"]:
                self.wins.append(1)
                self.losses.append(0)
                self.draws.append(0)
            elif reward == REWARDS["lose"]:
                self.wins.append(0)
                self.losses.append(1)
                self.draws.append(0)
            else:  # Draw
                self.wins.append(0)
                self.losses.append(0)
                self.draws.append(1)


            # saves the plot
            if (episode + 1) % 10000 == 0:
                print(self.draws.count(1))
                print(self.losses.count(1))
                print(self.wins.count(1))
                self.plot()

            # saves the stats and q table if future q tables are trash
            if (episode + 1) % 10000 == 0:
                self.save_stats()
                with open(f"q_table_{episode + 1}.pkl", "wb") as f:
                    pickle.dump(self.q_table, f)
                print(f"Saved Q-table at episode {episode + 1 + 0}")

        self.print_sorted()

    def save_stats(self, interval=10000):
        episode_number = len(self.episode_rewards)
        total_wins = sum(self.wins[-interval:])
        total_losses = sum(self.losses[-interval:])
        total_draws = sum(self.draws[-interval:])
        win_percentage = (total_wins / interval) * 100
        self.stats.append((episode_number, total_wins, total_losses, total_draws, win_percentage))

    def print_sorted(self):
        sorted_stats = sorted(self.stats, key=lambda x: x[4], reverse=True)
        print(sorted_stats)
        

    # plot can never be 100% cause it loses some gain at initial phrase
    def plot(self):
        episodes = np.arange(1, len(self.wins) + 1)
        cumulative_wins = np.cumsum(self.wins)
        win_rate = cumulative_wins / episodes * 100

        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        # Plot cumulative wins
        line1 = ax1.plot(episodes, cumulative_wins, 'b-', label='Cumulative Wins')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Cumulative Wins', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # Plot win rate
        line2 = ax2.plot(episodes, win_rate, 'g-', label='Win Rate (%)')
        ax2.set_ylabel('Win Rate (%)', color='g')
        ax2.tick_params(axis='y', labelcolor='g')

        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        plt.title(f'Training Progress (episodes = {len(self.wins)})')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'Training Progress({len(self.wins)+0}).png')


    def take_user_input(self):
        while True:
            user_input = input("Pick a point (matrix notation (0 indexing), e.g., 1,2 or 1 2): ").strip()
            parts = user_input.replace(',', ' ').split()
            if len(parts) == 2 and all(part.isdigit() for part in parts):
                x, y = map(int, parts)

                # Check if coordinates are valid
                if (x, y) in self.get_valid_move(self.game_board.board):
                    return (x, y)
                else:
                    print(f"Valid coords are {self.get_valid_move(self.game_board.board)}. Try again.")
            else:
                print("Invalid input format. Use 'x, y' or 'x y' with numbers.")


    def play(self):
        turn_of_player = input('Choose your turn (first or second): ').strip()
        self.game_board.reset()
        if turn_of_player.lower() == 'first':
            self.game_board.show_board_playable()
            while not self.game_board.done:           
                user_turn = self.take_user_input()
                next_state, _, winner = self.game_board.turn(user_turn, mark='x', playing=True)
                print(f"Your move {user_turn[0]}, {user_turn[1]}:")
                self.game_board.show_board_playable()

                if not self.game_board.done:
                    # ai turn
                    action = self.choose_action(next_state, playing=True)
                    q_values = self.get_q_values(next_state)
                    valid_move = self.get_valid_move(next_state)
                    valid_move_indices = self.get_valid_move_index(valid_move)
                    valid_q_values = q_values[valid_move_indices]
                    best_valid_move_idx = valid_move_indices[np.argmax(valid_q_values)]

                    print(f'State is: {next_state}\n')
                    print(f'Q Value: {q_values}\n')
                    print(f'Valid moves: {valid_move}\n')
                    print(f'Valid move index: {valid_move_indices}\n')
                    print(f'Q Table: {valid_q_values}')
                    print(f'Indices: {best_valid_move_idx}\n')
                    
                    print(f'Move selected: {self.total_moves[best_valid_move_idx]}\n')


                    next_state, _, winner = self.game_board.turn(action, mark='o', playing=True)
                    print(f"AI move {action[0]}, {action[1]}:")
                    self.game_board.show_board_playable()

                    if winner == 1:  # 'x' wins, aka player
                        print('Congrats, you win! Try training the bot with more episodes!')
                    elif winner == 2:  # 'o' wins, bot
                        print('Congrats, you lose! Maybe try to increase your skills loser, or decrease the episodes of the bot!')
                    elif winner == 0: #draw
                        print('It\'s a draw. Something never before seen in a game of tic-tac-toe.')
                elif winner == 1:  # 'x' wins, aka player
                    print('Congrats, you win! Try training the bot with more episodes!')
                elif winner == 2:  # 'o' wins, bot
                    print('Congrats, you lose! Maybe try to increase your skills loser, or decrease the episodes of the bot!')
                elif winner == 0: #draw
                    print('It\'s a draw. Something never before seen in a game of tic-tac-toe.')
        elif turn_of_player.lower() == 'second':
            while not self.game_board.done:
                # ai turn first
                state = self.game_board.board
                action = self.choose_action(state, playing=True)
                next_state, _, winner = self.game_board.turn(action, mark='x', playing=True)
                print(f"AI move {action[0]}, {action[1]}:")
                self.game_board.show_board_playable()

                if not self.game_board.done:
                    # user turn
                    user_turn = self.take_user_input()
                    next_state, _, winner = self.game_board.turn(user_turn, mark='o', playing=True)
                    print(f"Your move {user_turn[0]}, {user_turn[1]}:")
                    self.game_board.show_board_playable()

                    if winner == 1:  # 'x' wins, aka ai
                        print('Congrats, you lose! Maybe try to increase your skills loser, or decrease the episodes of the bot!')
                    elif winner == 2:  # 'o' wins, bot
                        print('Congrats, you win! Try training the bot with more episodes!')
                    elif winner == 0:  # draw
                        print('It\'s a draw. Something never before seen in a game of tic-tac-toe.')
                elif winner == 1:  # 'x' wins, aka ai
                    print('Congrats, you lose! Maybe try to increase your skills loser, or decrease the episodes of the bot!')
                elif winner == 2:  # 'o' wins, bot
                    print('Congrats, you win! Try training the bot with more episodes!')
                elif winner == 0:  # draw
                    print('It\'s a draw. Something never before seen in a game of tic-tac-toe.')
        else:
            self.play()


game = TicTacToe((3, 3))
with open('q_table_600000.pkl', 'rb') as f:
    q_table = pickle.load(f)
agent = QLearningAgent(game, epsilon=0,discount=0.9, lr=0.1, q_table=q_table) # for training, q_table=None
# agent.forward(600_000) #uncomment to train
agent.play()
