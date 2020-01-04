
import numpy, argparse
import torch
import torch.optim as optim
import copy, time

from model import Network
from game import MuZeroConfig, ReplayBuffer, make_janggi_config, Node
from mcts import expand_node, add_exploration_noise, run_mcts, select_action, play_game
from environ import Winner
################################################################################
############################# Testing the latest net ###########################
################################################################################

# Battle against random agents
def vs_random(network, config, n=20):
    results = {}
    for i in range(n):
        first_turn = i % 2 == 0
        turn = first_turn
        game = config.new_game()
        r = 0
        while not game.terminal():
            if turn:
              root = Node(0)
              current_observation = game.make_image(-1)
              legal = game.legal_actions()
              expand_node(root, game.to_play(), legal,
                          network.initial_inference(current_observation))
              add_exploration_noise(config, root)
              run_mcts(config, root, game.action_history(), network)
              action = select_action(config, len(game.history), root)
            else:
              action = numpy.random.choice(game.legal_actions())
            game.apply(action)
            turn = not turn
        if ((game.environment.winner == Winner.white and first_turn)
            or (game.environment.winner == Winner.black and not first_turn)):
          r = 1
        elif ((game.environment.winner == Winner.black and first_turn)
            or (game.environment.winner == Winner.white and not first_turn)):
          r = -1
        print(r)
        results[r] = results.get(r, 0) + 1
    return results

def random_vs_random(config, n=100):
    results = {}
    for i in range(n):
        first_turn = i % 2 == 0
        turn = first_turn
        game = config.new_game()
        r = 0
        while not game.terminal():
            action = numpy.random.choice(game.legal_actions())
            game.apply(action)
            turn = not turn
        if ((game.environment.winner == Winner.white and first_turn)
            or (game.environment.winner == Winner.black and not first_turn)):
          r = 1
        elif ((game.environment.winner == Winner.black and first_turn)
            or (game.environment.winner == Winner.white and not first_turn)):
          r = -1
        results[r] = results.get(r, 0) + 1
    return results

def latest_vs_older(last, old, config, n=20):
    results = {}
    for i in range(n):
        first_turn = i % 2 == 0
        turn = first_turn
        game = config.new_game()
        r = 0
        while not game.terminal():
            if turn:
              root = Node(0)
              current_observation = game.make_image(-1)
              legal = game.legal_actions()
              expand_node(root, game.to_play(), legal,
                          last.initial_inference(current_observation))
              add_exploration_noise(config, root)
              run_mcts(config, root, game.action_history(), last)
              action = select_action(config, len(game.history), root)
            else:
              root = Node(0)
              current_observation = game.make_image(-1)
              legal = game.legal_actions()
              expand_node(root, game.to_play(), legal,
                          old.initial_inference(current_observation))
              add_exploration_noise(config, root)
              run_mcts(config, root, game.action_history(), old)
              action = select_action(config, len(game.history), root)
            game.apply(action)
            turn = not turn
        if ((game.environment.winner == Winner.white and first_turn)
            or (game.environment.winner == Winner.black and not first_turn)):
          r = 1
        elif ((game.environment.winner == Winner.black and first_turn)
            or (game.environment.winner == Winner.white and not first_turn)):
          r = -1
        print(r)
        results[r] = results.get(r, 0) + 1
    return results

##### End Helpers ########
##########################
import tracemalloc

# MuZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def muzero(config: MuZeroConfig, device):
  network = Network(config.action_space_size, device).to(device)
  oldnet = copy.deepcopy(network)
  optimizer = optim.SGD(network.parameters(), lr=0.01, weight_decay=config.lr_decay_rate,
                        momentum=config.momentum)
  replay_buffer = ReplayBuffer(config)

  i = 0
  while True:
      t = time.time()
      game = play_game(config, network)
      replay_buffer.save_game(game)
      print('%d steps/s %.2f'%(i, game.environment.turn/(time.time()-t)))
      i += 1
      if i % config.checkpoint_interval == 0:
          batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
          update_weights(batch, network, optimizer, device)
          # Test against random agent
          vs_random_once = vs_random(network, config)
          print('network_vs_random = ', sorted(vs_random_once.items()), end='\n')
          vs_older = latest_vs_older(network, oldnet, device)
          print('lastnet_vs_older = ', sorted(vs_older.items()), end='\n')
          oldnet = copy.deepcopy(network)


def update_weights(batch, network, optimizer, device):

  network.train()    

  p_loss, v_loss = 0, 0

  for image, actions, targets in batch:
    # Initial step, from the real observation.
    value, reward, policy_logits, hidden_state = network.initial_inference(image)
    predictions = [(1.0, value, reward, policy_logits)]

    # Recurrent steps, from action and previous hidden state.
    for action in actions:
      value, reward, policy_logits, hidden_state = network.recurrent_inference(hidden_state, action)
      predictions.append((1.0 / len(actions), value, reward, policy_logits))

    for prediction, target in zip(predictions, targets):
      if(len(target[2]) > 0):
        _ , value, reward, policy_logits = prediction
        target_value, target_reward, target_policy = target

        p_loss += torch.sum(-torch.Tensor(numpy.array(target_policy)).to(device) * torch.log(policy_logits))
        v_loss += torch.sum((torch.Tensor([target_value]).to(device) - value) ** 2)
  
  optimizer.zero_grad()    
  total_loss = (p_loss + v_loss)
  total_loss.backward()
  optimizer.step()
  network.steps += 1
  print('p_loss %f v_loss %f' % (p_loss / len(batch), v_loss / len(batch)))

######### End Training ###########
##################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    args = parser.parse_args()
    device = 'cuda:0' if args.cuda else 'cpu'

    print(device)

    config = make_janggi_config()
    vs_random_once = random_vs_random(config)
    print('random_vs_random = ', sorted(vs_random_once.items()), end='\n')
    network = muzero(config, device)