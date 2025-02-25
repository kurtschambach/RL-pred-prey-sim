import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from datetime import *

# === Constants & Setup ===
WIDTH, HEIGHT = 800, 800
FPS = 10
NUMBER_DOTS = 50
HIDDEN_LAYER_N = 16    # Number of Neurons in the Hidden Layers
EPISODE_LENGTH = 1000   # Frames per episode
NUM_EPISODES = 30000   # Total training episodes
GAMMA = 0.999          # Discount factor for returns
RUN_TITLE = f"run43-{HIDDEN_LAYER_N}-{NUMBER_DOTS / 100}d-{EPISODE_LENGTH / 100}f-n017-std"    # Name of run for TensorBoard Stats run<count>-<num NN dim>-<num dots 100>-<frames per episode 100>-<feat>
DARK = True
PARAM_SEARCH = False   # If true running until some param gets good enough -> check has to be implemented

# Colors (RGB)
WHITE = (230, 220, 220)
BLACK = (20, 20, 20)
GREEN = (0, 255, 160)
RED   = (255, 80, 100)
BLUE  = (70, 120, 255)
agent_colors = [GREEN, BLUE] # TODO: custom number of agents

# Field of view settings for fog-of-war (in pixels)
VISION_RANGE = 150  # Radius of the area visible to each prey

# === Neural Network Policy for Agents ===
# This network takes a 6-dimensional state vector and outputs a 2-dimensional action.
class SimplePolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimplePolicy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_LAYER_N),
            nn.Linear(HIDDEN_LAYER_N, HIDDEN_LAYER_N),
            nn.Linear(HIDDEN_LAYER_N, output_dim)
        )
        # Initialize a learnable parameter for log_std.
        self.log_std = nn.Parameter(torch.tensor([-0.17 for _ in range(output_dim)]))  # Starts with std = exp(0) = 1.

    def forward(self, x):
        mean = self.fc(x)
        std = torch.exp(self.log_std)  # Ensure std is positive.
        return mean, std

# --- Helper functions ---
def latestAvgGreater(list: list, distribution: int, treshold: int):
    if len(list) < treshold: return False
    list.reverse()
    if (sum(list[:distribution]) / distribution) < treshold: return False  # TODO: check code
    return True

# === Helper Functions for Wall Collisions ===
def circle_rect_collision(cx, cy, radius, rect):
    """
    Check collision between a circle (centered at (cx, cy) with given radius)
    and a rectangle (pygame.Rect).
    """
    # Find the closest point on the rectangle to the circle's center.
    closest_x = max(rect.left, min(cx, rect.right))
    closest_y = max(rect.top, min(cy, rect.bottom))
    distance = np.hypot(cx - closest_x, cy - closest_y)
    return distance < radius

def check_collision_with_walls(x, y, radius, walls):
    """
    Returns True if a circle at (x, y) with the given radius collides with any wall.
    """
    for wall in walls:
        if circle_rect_collision(x, y, radius, wall.rect):
            return True
    return False

# === Base Agent Class ===
class Agent:
    def __init__(self, x, y, color=BLUE):
        self.x = x
        self.y = y
        self.radius = 10
        self.color = color
        self.vel = 5
        self.xp = 0  # Experience points / reward
        self.target = (0, 0)
    def draw(self, screen):
        pygame.draw.aaline(screen, WHITE if DARK else BLACK, (int(self.x), int(self.y)), self.target)
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

# === Prey Agent Class with RL ===
class PreyAgent(Agent):
    def __init__(self, x, y, agent_count):
        super().__init__(x, y, color=agent_colors[agent_count%3])
        # Policy network: 6-dimensional input -> 2-dimensional output (action mean)
        self.model = SimplePolicy(input_dim=6, output_dim=2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        # Buffer to store (log_prob, reward) tuples for the episode
        self.buffer = []
        # For reward calculation: track previous xp value
        self.prev_xp = 0.0

    def get_state(self, predator, dots):
        """
        Create a state vector:
          - Normalized own position: (x/WIDTH, y/HEIGHT)
          - Relative position to nearest green dot (normalized)
          - Relative position to the predator (normalized)
        """
        norm_x = self.x / WIDTH
        norm_y = self.y / HEIGHT

        # Find the nearest green dot.
        green_dots = [dot for dot in dots if dot.type == 'green']
        if green_dots:
            nearest_dot = min(green_dots, key=lambda d: np.hypot(self.x - d.x, self.y - d.y))
            self.target = (int(nearest_dot.x), int(nearest_dot.y))
            dx_dot = (nearest_dot.x - self.x) / WIDTH
            dy_dot = (nearest_dot.y - self.y) / HEIGHT
        else:
            dx_dot, dy_dot = 0.0, 0.0

        # Relative position to predator.
        dx_pred = (predator.x - self.x) / WIDTH
        dy_pred = (predator.y - self.y) / HEIGHT

        state = np.array([norm_x, norm_y, dx_dot, dy_dot, dx_pred, dy_pred], dtype=np.float32)
        return state

    def select_action(self, predator, dots, walls, writer, episode):
        """
        Use the policy network to select an action.
        Returns the state and the log probability of the action.
        """
        state = self.get_state(predator, dots)
        state_tensor = torch.from_numpy(state).unsqueeze(0)  # shape: (1, input_dim)
        mean, std = self.model.forward(state_tensor)  # Now you get both mean and std.
        writer.add_scalar('Params/std', std[0], global_step=episode)
        dist = distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        # (Optional) Normalize the action:
        action_np = action.detach().numpy()[0]
        norm = np.linalg.norm(action_np) + 1e-8  # Adding epsilon to avoid division by zero.
        action_np = action_np / norm
        norm = np.linalg.norm(action_np)
        if norm != 0:
            action_np = action_np / norm
        # Attempt movement (axis-wise movement with wall collision checking)
        new_x = self.x + action_np[0] * self.vel
        if not check_collision_with_walls(new_x, self.y, self.radius, walls):
            self.x = new_x
        new_y = self.y + action_np[1] * self.vel
        if not check_collision_with_walls(self.x, new_y, self.radius, walls):
            self.y = new_y
        # Clamp position to keep agent within the screen
        self.x = max(self.radius, min(WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(HEIGHT - self.radius, self.y))
        return state, log_prob

    def store_reward(self, reward):
        """Store the most recent reward (to be paired with the corresponding log_prob stored in buffer)."""
        # Append reward to the last stored tuple (log_prob, reward) in the buffer.
        # Here, we assume that we call this right after select_action so that the last log_prob corresponds.
        if self.buffer:
            log_prob, _ = self.buffer[-1]
            self.buffer[-1] = (log_prob, reward)
        else:
            self.buffer.append((None, reward))  # Should not happen

    def record_step(self, log_prob, reward):
        """Record a step in the buffer."""
        self.buffer.append((log_prob, reward))

    def update_policy(self, writer, episode, gamma=GAMMA):
        """Perform a REINFORCE update on the policy network using the episode buffer."""
        R = 0
        policy_loss = []
        returns = []
        # Compute the discounted return for each frame (in reverse)
        for (_, r) in reversed(self.buffer):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        # Normalize returns for more stable updates.
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        for (log_prob, _), G in zip(self.buffer, returns):
            # Skip if log_prob is None (should not happen)
            if log_prob is not None:
                policy_loss.append(-log_prob * G)
        if policy_loss:
            self.optimizer.zero_grad()
            loss = torch.cat(policy_loss).sum()
            writer.add_scalar('Loss/Policy', loss or -1, global_step=episode)
            loss.backward()
            self.optimizer.step()
        # Clear buffer after update
        self.buffer = []
        # Reset previous xp tracker.
        self.prev_xp = self.xp

# === Predator Agent Class ===
class PredatorAgent(Agent):
    def __init__(self, x, y):
        super().__init__(x, y, color=RED)
        self.waiting = -1
        self.vel = 3
    def update(self, target, walls):
        # Simple chasing behavior: move toward the target prey.
        if (self.waiting == -1):
            self.target = (int(target.x), int(target.y))
            dx = target.x - self.x
            dy = target.y - self.y
            norm = np.sqrt(dx**2 + dy**2)
            if norm != 0:
                dx, dy = dx / norm, dy / norm

            # Attempt axis-wise movement with wall collision checking.
            new_x = self.x + dx * self.vel
            if not check_collision_with_walls(new_x, self.y, self.radius, walls):
                self.x = new_x

            new_y = self.y + dy * self.vel
            if not check_collision_with_walls(self.x, new_y, self.radius, walls):
                self.y = new_y

            # Clamp positions so the predator stays within the screen.
            self.x = max(self.radius, min(WIDTH - self.radius, self.x))
            self.y = max(self.radius, min(HEIGHT - self.radius, self.y))

# === Dot Class (Reward or Penalty Items) ===
class Dot:
    def __init__(self, x, y, dot_type='green'):
        self.x = x
        self.y = y
        self.radius = 5
        self.type = dot_type  # 'green' for XP reward, 'red' for XP penalty
        self.color = GREEN if dot_type == 'green' else RED
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

# === Wall Class ===
class Wall:
    def __init__(self, rect):
        self.rect = rect  # A pygame.Rect object
        self.color = WHITE if DARK else BLACK
    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)

# === Collision Helper for Circles (Agents/Dots) ===
def check_collision_circle(a, b):
    """Return True if two circular objects (agents or a dot and an agent) overlap."""
    dist = np.hypot(a.x - b.x, a.y - b.y)
    return dist < (a.radius + b.radius)

# === Parameter Exchange Function (for Prey Agents) ===
def exchange_parameters(agent1: PreyAgent, agent2: PreyAgent, alpha=0.5):
    """
    Blends the parameters of the two agents' networks.
    The new parameters are an average weighted by alpha.
    TODO: fix func description, only for agent with less xp (but change that too, take avg)
    """
    with torch.no_grad():
        if agent1.xp > agent2.xp:
            for param1, param2 in zip(agent1.model.parameters(), agent2.model.parameters()):
                new_param = alpha * param1.data + (1 - alpha) * param2.data
                param2.data.copy_(new_param)
        else:
            for param1, param2 in zip(agent1.model.parameters(), agent2.model.parameters()):
                new_param = alpha * param2.data + (1 - alpha) * param1.data
                param1.data.copy_(new_param)

# === Environment Reset Function ===
def reset_environment(prey_agents, predator_agent, dots):
    """Reset positions of agents and regenerate dots for a new episode."""
    # Reset prey positions and XP
    prey_agents[0].x, prey_agents[0].y = 100, 100
    prey_agents[0].xp = 0
    prey_agents[0].prev_xp = 0
    prey_agents[0].buffer = []

    prey_agents[1].x, prey_agents[1].y = 700, 700
    prey_agents[1].xp = 0
    prey_agents[1].prev_xp = 0
    prey_agents[1].buffer = []

    # Reset predator position and XP
    predator_agent.x, predator_agent.y = 400, 400
    predator_agent.xp = 0

    # # Regenerate dots
    # dots.clear()
    # for _ in range(NUMBER_DOTS):
    #     x = random.randint(20, WIDTH - 20)
    #     y = random.randint(20, HEIGHT - 20)
    #     dot_type = 'green' if random.random() < 1.0 else 'red'
    #     dots.append(Dot(x, y, dot_type))


# === Main Simulation Loop ===
def main():
    pygame.init()
    fps = FPS
    episode_length = EPISODE_LENGTH
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    render_all = True
    settings_changed = False
    pygame.display.set_caption("Predator-Prey Simulation")
    writer = SummaryWriter(log_dir=f"runs/predator-prey-simulation/{RUN_TITLE}")
    clock = pygame.time.Clock()
    frame_count = 0

    # --- Create Walls (as pygame.Rect objects) ---
    walls = [
        # Wall(pygame.Rect(200, 200, 400, 20)),  # Horizontal wall
        # Wall(pygame.Rect(200, 580, 400, 20)),  # Another horizontal wall
        # Wall(pygame.Rect(200, 220, 20, 360))   # Vertical wall (optional)
    ]

    # --- Create Dots ---
    dots = []
    for _ in range(NUMBER_DOTS):
        x = random.randint(20, WIDTH - 20)
        y = random.randint(20, HEIGHT - 20)
        dot_type = 'green' if random.random() < 0.7 else 'red'
        dots.append(Dot(x, y, dot_type))

    # --- Create Agents ---
    prey_agents = [PreyAgent(100, 100, 0), PreyAgent(700, 700, 1)]
    predator_agent = PredatorAgent(400, 400)

    # --- Meeting/Exchange Settings for Prey ---
    meeting_interval = 1000  # Check for meeting every 300 episodes
    meeting_threshold = 50  # Prey must be within 50 pixels to "meet"

    # --- Setup Matplotlib for Real-Time XP Graph with Two Lines ---
    plt.style.use('dark_background')  # alt: 'Solarize_Light2'
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()

    x_data = []       # Episode count
    x_data_smooth = []    # Episode % 10 count
    prey_a_data = []    # XP for prey agent a
    prey_b_data = []    # XP for prey agent b
    pred_data = []    # XP for the predator agent
    prey_avg_smooth_data = []    # Smoothed Avg XP for prey agents
    efficiency = []    # Prey Efficiency
    smooth_avg_collector = []    # Collect Avg Xp before adding to plot

    # Create line objects with labels
    line_efficiency, = ax.plot(x_data, efficiency, '#7010F0', label="Prey Effieciency")
    line_prey_a, = ax.plot(x_data, prey_a_data, '-g', label="Prey A XP")
    line_prey_b, = ax.plot(x_data, prey_b_data, '-b', label="Prey B XP")
    line_pred, = ax.plot(x_data, pred_data, '-r', label="Predator XP * 0.05")
    line_smooth_avg, = ax.plot(x_data, prey_avg_smooth_data, '-m', label="Prey Smoothed Avg XP")

    ax.legend()
    ax.set_title("XP Over Time")
    ax.set_xlabel("Episode")
    ax.set_ylabel("XP")

    episode = 0
    running = True
    while running and (PARAM_SEARCH or episode < NUM_EPISODES):
        frame_count = 0
        # Reset environment at the start of each episode.
        reset_environment(prey_agents, predator_agent, dots)


        while frame_count < episode_length:
            clock.tick(fps)
            frame_count += 1

            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if ((event.key == pygame.K_UP) and (fps < 10000)):
                        fps = fps * 10
                        settings_changed = True
                    if ((event.key == pygame.K_DOWN) and (fps > 1)):
                        fps = fps / 10
                        settings_changed = True
                    if (event.key == pygame.K_SPACE):
                        render_all = not render_all
                        settings_changed = True

            # --- Prey Agents select actions and move ---
            for prey in prey_agents:
                # Before moving, record current xp for reward computation.
                old_xp = prey.xp
                state, log_prob = prey.select_action(predator_agent, dots, walls, writer, episode)
                # We record the step; reward will be computed after collision processing.
                # For now, we store the (log_prob, placeholder) in the buffer.
                prey.record_step(log_prob, 0)  # reward will be updated below.

            # --- Predator Agent Update ---
            # Predator chases the closest prey.
            target_prey = min(prey_agents, key=lambda p: np.hypot(predator_agent.x - p.x, predator_agent.y - p.y))
            predator_agent.update(target_prey, walls)

            # --- Waiting Time Check for Predator
            #if ((predator_agent.waiting != -1) and (predator_agent.waiting < frame_count)): predator_agent.waiting = -1

            # --- Dot Collision Check for Prey ---
            for prey in prey_agents:
                for dot in dots[:]:
                    if check_collision_circle(prey, dot):
                        dots.remove(dot)  # Remove dot after collision
                        x = random.randint(20, WIDTH - 20)  # Add a dot
                        y = random.randint(20, HEIGHT - 20)
                        if dot.type == 'green':
                            prey.xp += 1   # Gain XP
                            dots.append(Dot(x, y, 'green'))
                        else:
                            prey.xp -= 0.5 # Lose XP
                            dots.append(Dot(x, y, 'red'))

            # --- Predator Catches Prey ---
            for prey in prey_agents:
                if check_collision_circle(predator_agent, prey):
                    #if prey.xp > -5: prey.xp -= 5  # Penalty for being caught
                    predator_agent.waiting = frame_count + 300
                    predator_agent.xp += 5

            # # --- Prey Meeting and Parameter Exchange ---
            # if (episode - 1) % meeting_interval == 0:
            #     if np.hypot(prey_agents[0].x - prey_agents[1].x,
            #                 prey_agents[0].y - prey_agents[1].y) < meeting_threshold:
            #         # Blend their network parameters.
            #         exchange_parameters(prey_agents[0], prey_agents[1], alpha=0.3)

            # Now, update the reward in the buffer for each prey agent
            for prey in prey_agents:
                # Compute reward as change in xp since last frame.
                reward = prey.xp - prey.prev_xp
                # Update the last stored frame (overwrite the placeholder reward).
                if prey.buffer:
                    log_prob, _ = prey.buffer[-1]
                    prey.buffer[-1] = (log_prob, reward)
                prey.prev_xp = prey.xp

            if render_all:
                # --- Rendering the whole Environment ---
                screen.fill(BLACK if DARK else BLACK if DARK else WHITE)
                for wall in walls:
                    wall.draw(screen)
                for dot in dots:
                    dot.draw(screen)
                for prey in prey_agents:
                    prey.draw(screen)
                predator_agent.draw(screen)

                # --- Fog of War Overlay ---
                # Create a fog surface that darkens the entire screen.
                fog = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                fog.fill((0, 0, 0, 100))  # Outside the field-of-view, fog is ~78% opaque.
                # For each prey, draw a circle representing its field-of-view with lighter fog (~20% opacity).
                for prey in prey_agents:
                    pygame.draw.circle(fog, (0, 0, 0, 51), (int(prey.x), int(prey.y)), VISION_RANGE)
                screen.blit(fog, (0, 0))

                # --- Display XP Values for Prey ---
                font = pygame.font.SysFont("Arial", 18)
                episode_text = font.render(f"Episode: {episode}", True, WHITE if DARK else BLACK, BLACK if DARK else WHITE)
                fps_text = font.render(f"FPS: {fps}", True, WHITE if DARK else BLACK, BLACK if DARK else WHITE)
                screen.blit(episode_text, (10, 10))
                screen.blit(fps_text, (10, 30))
                for idx, prey in enumerate(prey_agents):
                    xp_text = font.render(f"Prey {idx+1} XP: {prey.xp:.1f}", True, WHITE if DARK else BLACK, BLACK if DARK else WHITE)
                    screen.blit(xp_text, (10, 50 + idx * 20))

                pygame.display.flip()
            elif settings_changed:
                settings_changed = False
                font = pygame.font.SysFont("Arial", 18)
                info_text = font.render("Paused Real-Time-Simulation", True, BLACK if DARK else WHITE, WHITE if DARK else BLACK)
                fps_text = font.render(f"FPS: {fps}", True, BLACK if DARK else WHITE, WHITE if DARK else BLACK)
                screen.blit(info_text, (10, 10))
                screen.blit(fps_text, (10, 30))

                pygame.display.flip()

        # --- End of Episode: Update the Policy for Each Prey Agent ---
        for prey in prey_agents:
            prey.update_policy(writer=writer, episode=episode, gamma=GAMMA)

        # Append new data points.
        prey_avg = sum(prey.xp for prey in prey_agents) / len(prey_agents) if prey_agents else 0
        prey_eff = (prey_avg / episode_length) / NUMBER_DOTS
        x_data.append(episode)
        prey_a_data.append(prey_agents[0].xp)
        prey_b_data.append(prey_agents[1].xp)
        pred_data.append(predator_agent.xp / 20)
        efficiency.append(prey_eff)
        smooth_avg_collector.append(prey_avg)
        if episode % round(episode_length / 10) == 0:
            x_data_smooth.append(episode)
            prey_avg_smooth_data.append((sum(smooth_avg_collector) / len(smooth_avg_collector)))
            smooth_avg_collector = []

        if PARAM_SEARCH and efficiency[-1] > 5:
            print("Try episode length increase")
            if latestAvgGreater(efficiency, 10, treshold=1.5):  # Increase episode_length if efficiency avg over past 10 episodes > 2
                episode_length *= 2
                print(f"\n--- Set episode length to {episode_length} frames at {datetime.now()} ---\n")

        # Update TensorBoard
        writer.add_scalar('Metrics/AvgPreyXP', prey_avg, global_step=episode)
        writer.add_scalar('Metrics/PreyEfficiency', prey_eff, global_step=episode)

        # Update line objects.
        line_prey_a.set_xdata(x_data)
        line_prey_a.set_ydata(prey_a_data)
        line_prey_b.set_xdata(x_data)
        line_prey_b.set_ydata(prey_b_data)
        line_pred.set_xdata(x_data)
        line_pred.set_ydata(pred_data)
        line_efficiency.set_xdata(x_data)
        line_efficiency.set_ydata(efficiency)
        line_smooth_avg.set_xdata(x_data_smooth)
        line_smooth_avg.set_ydata(prey_avg_smooth_data)

        # Update

        # Rescale the axes to accommodate the new data.
        ax.relim()
        ax.autoscale_view()

        # Redraw the figure.
        fig.canvas.draw()
        fig.canvas.flush_events()

        # A short pause to allow the plot to update.
        plt.pause(0.001)

        episode += 1
        if not PARAM_SEARCH: print(f"Episode {episode} finished. Avg Prey XP: {prey_avg:.2f}")

    pygame.quit()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
