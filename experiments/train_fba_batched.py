"""
FBA-0 batched training with PPO + GAE + movement-direction auxiliary.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import time
from pathlib import Path
from dataclasses import dataclass

from experiments.env_cnw import (
    CNWConfig, generate_world, reset, step, get_observation,
    obs_to_onehot, obs_dim, N_ACTIONS,
)
from experiments.arch_fba import make_agent, count_params


@dataclass
class TrainConfig:
    arch: str = "fba"
    seed: int = 0
    n_updates: int = 5000
    batch_size: int = 32
    max_steps: int = 100
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    pred_coef: float = 0.5
    value_coef: float = 0.5
    delta_coef: float = 1.0
    max_grad_norm: float = 0.5
    bptt: bool = True
    n_train_worlds: int = 200
    n_eval_worlds: int = 50
    eval_interval: int = 200
    log_interval: int = 50
    checkpoint_interval: int = 1000
    output_dir: str = "experiments/results/fba_0"
    obs_noise: float = 0.30
    obs_perm: bool = True
    action_perm: bool = True


class BatchedEnv:
    def __init__(self, worlds, config, rng, batch_size):
        self.worlds = worlds
        self.config = config
        self.rng = rng
        self.batch_size = batch_size
        self.states = [None] * batch_size
        self.current_goals = [0] * batch_size
        self.active_worlds = [None] * batch_size

    def reset_all(self, goal_indices=None):
        obs_batch = []
        for i in range(self.batch_size):
            world = self.worlds[self.rng.integers(len(self.worlds))]
            self.active_worlds[i] = world
            self.states[i] = reset(world, self.rng)
            if goal_indices is not None:
                self.current_goals[i] = goal_indices[i]
            else:
                self.current_goals[i] = int(self.rng.integers(self.config.n_landmarks))
            obs_raw = get_observation(self.states[i], self.rng)
            obs_batch.append(obs_to_onehot(obs_raw))
        return np.stack(obs_batch), np.array(self.current_goals)

    def step_all(self, actions):
        obs_batch = []
        rewards = []
        dones = []
        deltas = []
        for i in range(self.batch_size):
            obs_vec, reward, done, info = step(
                self.states[i], int(actions[i]), self.rng,
                goal_idx=self.current_goals[i]
            )
            obs_batch.append(obs_vec)
            rewards.append(reward)
            dones.append(done)
            deltas.append(info["delta_class"])
        return np.stack(obs_batch), np.array(rewards), np.array(dones), np.array(deltas)

    def get_successes(self):
        return [self.current_goals[i] in self.states[i].goals_reached
                for i in range(self.batch_size)]


@torch.no_grad()
def collect_rollout(agent, env, device):
    B = env.batch_size
    obs_np, goals_np = env.reset_all()

    goals_t = torch.tensor(goals_np, dtype=torch.long, device=device)
    obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
    prev_actions = torch.zeros(B, dtype=torch.long, device=device)
    agent_state = agent.initial_state(B, device)

    all_obs = []
    all_actions = []
    all_old_log_probs = []
    all_values = []
    all_rewards = []
    all_masks = []
    all_dones = []
    all_pred_targets = []
    all_deltas = []

    active = np.ones(B, dtype=bool)

    for t in range(env.config.max_steps):
        all_obs.append(obs_t)

        logits, pred, value, agent_state, extras = agent(
            obs_t, prev_actions, goals_t, agent_state, training=False
        )

        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()

        all_actions.append(actions)
        all_old_log_probs.append(dist.log_prob(actions))
        all_values.append(value)

        obs_np, rewards_np, dones_np, deltas_np = env.step_all(actions.cpu().numpy())
        obs_next = torch.tensor(obs_np, dtype=torch.float32, device=device)

        mask = torch.tensor(active.copy(), dtype=torch.float32, device=device)
        all_masks.append(mask)
        all_rewards.append(torch.tensor(rewards_np, dtype=torch.float32, device=device))
        all_dones.append(torch.tensor(dones_np.copy(), dtype=torch.bool, device=device))
        all_pred_targets.append(obs_next)
        all_deltas.append(torch.tensor(deltas_np, dtype=torch.long, device=device))

        active &= ~dones_np
        if not active.any():
            break

        obs_t = obs_next
        prev_actions = actions

        if agent_state:
            for i in range(B):
                if dones_np[i]:
                    for k in agent_state:
                        agent_state[k][i] = 0.0

    return {
        "obs": torch.stack(all_obs),
        "actions": torch.stack(all_actions),
        "old_log_probs": torch.stack(all_old_log_probs),
        "values": torch.stack(all_values),
        "rewards": torch.stack(all_rewards),
        "masks": torch.stack(all_masks),
        "dones": torch.stack(all_dones),
        "pred_targets": torch.stack(all_pred_targets),
        "deltas": torch.stack(all_deltas),
        "goals": goals_t,
        "successes": env.get_successes(),
    }


def compute_gae(rewards, values, masks, dones, gamma, lam):
    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = torch.zeros(B, device=rewards.device)
        else:
            next_value = values[t + 1]

        next_non_terminal = (~dones[t]).float()
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        last_gae = last_gae * masks[t]
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def ppo_epoch(agent, rollout, advantages, returns, optimizer, cfg, device):
    T = rollout["obs"].shape[0]
    B = rollout["obs"].shape[1]

    state = agent.initial_state(B, device)
    prev_act = torch.zeros(B, dtype=torch.long, device=device)

    new_log_probs = []
    new_values = []
    new_entropies = []
    new_pred_losses = []
    new_delta_losses = []

    for t in range(T):
        if not cfg.bptt:
            state = {k: v.detach() for k, v in state.items()} if state else {}
        logits, pred, value, state, extras = agent(
            rollout["obs"][t], prev_act, rollout["goals"], state, training=True
        )

        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs.append(dist.log_prob(rollout["actions"][t]))
        new_values.append(value)
        new_entropies.append(dist.entropy())

        pred_loss = ((pred - rollout["pred_targets"][t]) ** 2).mean(dim=-1)
        new_pred_losses.append(pred_loss)

        delta_logits = extras.get("delta_logits")
        if delta_logits is not None:
            dl = F.cross_entropy(delta_logits, rollout["deltas"][t], reduction='none')
            new_delta_losses.append(dl)

        if state and rollout["dones"][t].any():
            alive = (~rollout["dones"][t]).float().unsqueeze(-1)
            state = {k: v * alive for k, v in state.items()}

        prev_act = rollout["actions"][t]

    new_log_probs = torch.stack(new_log_probs)
    new_values = torch.stack(new_values)
    new_entropies = torch.stack(new_entropies)
    new_pred_losses = torch.stack(new_pred_losses)
    masks = rollout["masks"]

    adv = advantages.detach()
    adv_valid = adv[masks > 0]
    if adv_valid.numel() > 1:
        adv = (adv - adv_valid.mean()) / (adv_valid.std() + 1e-8)

    ratio = torch.exp(new_log_probs - rollout["old_log_probs"])
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
    policy_loss = -(torch.min(surr1, surr2) * masks).sum() / masks.sum()

    value_loss = ((new_values - returns.detach()) ** 2 * masks).sum() / masks.sum()
    pred_loss = (new_pred_losses * masks).sum() / masks.sum()
    entropy = (new_entropies * masks).sum() / masks.sum()

    loss = (policy_loss
            + cfg.value_coef * value_loss
            + cfg.pred_coef * pred_loss
            - cfg.entropy_coef * entropy)

    if new_delta_losses:
        delta_losses = torch.stack(new_delta_losses)
        delta_loss = (delta_losses * masks).sum() / masks.sum()
        loss = loss + cfg.delta_coef * delta_loss
    else:
        delta_loss = torch.tensor(0.0)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
    optimizer.step()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "pred_loss": pred_loss.item(),
        "delta_loss": delta_loss.item(),
        "entropy": entropy.item(),
        "clip_frac": ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item(),
    }


def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = "cpu"

    env_config = CNWConfig(
        obs_noise=cfg.obs_noise,
        obs_perm_enabled=cfg.obs_perm,
        action_perm_enabled=cfg.action_perm,
    )
    train_worlds = [generate_world(env_config, rng) for _ in range(cfg.n_train_worlds)]
    eval_worlds = [generate_world(env_config, rng) for _ in range(cfg.n_eval_worlds)]

    agent = make_agent(cfg.arch)
    agent.to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    out_dir = Path(cfg.output_dir) / f"{cfg.arch}_s{cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log = {
        "arch": cfg.arch, "seed": cfg.seed,
        "params": count_params(agent), "algorithm": "ppo",
        "obs_noise": cfg.obs_noise, "obs_perm": cfg.obs_perm,
        "action_perm": cfg.action_perm, "bptt": cfg.bptt,
        "delta_coef": cfg.delta_coef,
        "train_curve": [], "eval_single": [], "eval_compose": [],
    }

    train_env = BatchedEnv(train_worlds, env_config, rng, cfg.batch_size)

    running_reward = 0.0
    running_success = 0.0
    total_episodes = 0
    t0 = time.time()

    for update in range(cfg.n_updates):
        agent.eval()
        rollout = collect_rollout(agent, train_env, device)
        agent.train()

        advantages, returns = compute_gae(
            rollout["rewards"], rollout["values"],
            rollout["masks"], rollout["dones"],
            cfg.gamma, cfg.gae_lambda,
        )

        epoch_stats = []
        for _ in range(cfg.ppo_epochs):
            stats = ppo_epoch(agent, rollout, advantages, returns,
                              optimizer, cfg, device)
            epoch_stats.append(stats)

        avg_loss = np.mean([s["loss"] for s in epoch_stats])
        avg_clip = np.mean([s["clip_frac"] for s in epoch_stats])
        avg_entropy = np.mean([s["entropy"] for s in epoch_stats])
        avg_delta = np.mean([s["delta_loss"] for s in epoch_stats])

        batch_reward = (rollout["rewards"] * rollout["masks"]).sum(dim=0).mean().item()
        batch_success = np.mean(rollout["successes"])
        running_reward = 0.95 * running_reward + 0.05 * batch_reward
        running_success = 0.95 * running_success + 0.05 * batch_success
        total_episodes += cfg.batch_size

        if (update + 1) % cfg.log_interval == 0:
            elapsed = time.time() - t0
            eps_per_sec = total_episodes / elapsed
            log["train_curve"].append({
                "update": update + 1,
                "episodes": total_episodes,
                "running_reward": round(running_reward, 4),
                "running_success": round(running_success, 4),
                "loss": round(avg_loss, 4),
                "delta_loss": round(avg_delta, 4),
                "clip_frac": round(avg_clip, 4),
                "entropy": round(avg_entropy, 4),
                "eps_per_sec": round(eps_per_sec, 1),
            })
            print(f"[{cfg.arch} s{cfg.seed}] u={update+1:5d} ep={total_episodes:6d} "
                  f"r={running_reward:.3f} succ={running_success:.3f} "
                  f"dl={avg_delta:.3f} ent={avg_entropy:.3f} "
                  f"clip={avg_clip:.2f} ({eps_per_sec:.1f} ep/s)")

        if (update + 1) % cfg.eval_interval == 0:
            single_rate, compose_rate, per_goal = evaluate(
                agent, eval_worlds, env_config, rng, device
            )
            log["eval_single"].append({
                "update": update + 1, "episodes": total_episodes,
                "rate": round(float(single_rate), 4),
                "per_goal": [round(float(x), 4) for x in per_goal],
            })
            log["eval_compose"].append({
                "update": update + 1, "episodes": total_episodes,
                "rate": round(float(compose_rate), 4),
            })
            print(f"  EVAL single={single_rate:.3f} "
                  f"(per_goal={[f'{x:.2f}' for x in per_goal]}) "
                  f"compose={compose_rate:.3f}")
            agent.train()

        if (update + 1) % cfg.checkpoint_interval == 0:
            torch.save({
                "model_state": agent.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "update": update + 1, "log": log,
            }, out_dir / f"ckpt_{update+1}.pt")

    single_rate, compose_rate, per_goal = evaluate(
        agent, eval_worlds, env_config, rng, device
    )
    log["final_single_rate"] = round(float(single_rate), 4)
    log["final_compose_rate"] = round(float(compose_rate), 4)
    log["final_per_goal"] = [round(float(x), 4) for x in per_goal]
    log["total_time_s"] = round(time.time() - t0, 1)
    log["total_episodes"] = total_episodes

    torch.save(agent.state_dict(), out_dir / "final_model.pt")
    with open(out_dir / "log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n[{cfg.arch} s{cfg.seed}] DONE in {log['total_time_s']:.0f}s")
    print(f"  Final single={log['final_single_rate']:.3f} "
          f"compose={log['final_compose_rate']:.3f}")
    return log


def evaluate(agent, eval_worlds, env_config, rng, device):
    agent.eval()
    with torch.no_grad():
        per_goal_rates = []
        all_single = []
        for g in range(env_config.n_landmarks):
            goal_successes = []
            for w in eval_worlds:
                success = run_single_eval(agent, w, g, rng, device, env_config.max_steps)
                goal_successes.append(float(success))
                all_single.append(float(success))
            per_goal_rates.append(np.mean(goal_successes))
        single_rate = np.mean(all_single)

        compose_successes = []
        for w in eval_worlds:
            success = run_compose_eval(agent, w, rng, device, env_config)
            compose_successes.append(float(success))
        compose_rate = np.mean(compose_successes)

    return single_rate, compose_rate, per_goal_rates


def run_single_eval(agent, world, goal_idx, rng, device, max_steps):
    from experiments.env_cnw import reset as env_reset, get_observation as env_obs
    state = env_reset(world, rng)
    obs_raw = env_obs(state, rng)
    obs_vec = obs_to_onehot(obs_raw)
    obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
    prev_act = torch.zeros(1, dtype=torch.long, device=device)
    goal_t = torch.tensor([goal_idx], dtype=torch.long, device=device)
    a_state = agent.initial_state(1, device)

    for t in range(max_steps):
        logits, _, _, a_state, _ = agent(obs_t, prev_act, goal_t, a_state, training=False)
        action = logits.argmax(dim=-1)
        obs_vec, reward, done, info = step(state, action.item(), rng, goal_idx=goal_idx)
        if goal_idx in state.goals_reached:
            return True
        if done:
            return False
        obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
        prev_act = action
    return False


def run_compose_eval(agent, world, rng, device, env_config):
    from experiments.env_cnw import reset as env_reset, get_observation as env_obs
    state = env_reset(world, rng)
    obs_raw = env_obs(state, rng)
    obs_vec = obs_to_onehot(obs_raw)
    obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
    prev_act = torch.zeros(1, dtype=torch.long, device=device)
    a_state = agent.initial_state(1, device)

    current_goal = 0
    for t in range(env_config.max_steps):
        goal_t = torch.tensor([current_goal], dtype=torch.long, device=device)
        logits, _, _, a_state, _ = agent(obs_t, prev_act, goal_t, a_state, training=False)
        action = logits.argmax(dim=-1)
        obs_vec, reward, done, info = step(state, action.item(), rng, goal_idx=current_goal)

        if state.agent_pos == world.landmark_positions[current_goal]:
            current_goal += 1
            if current_goal >= env_config.n_landmarks:
                return True
        if done:
            return False
        obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
        prev_act = action
    return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="fba",
                        choices=["fba", "flat", "cross", "reactive"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="experiments/results/fba_0")
    parser.add_argument("--obs-noise", type=float, default=0.0)
    parser.add_argument("--obs-perm", action="store_true", default=False)
    parser.add_argument("--action-perm", action="store_true", default=False)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--delta-coef", type=float, default=1.0)
    parser.add_argument("--no-bptt", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    cfg = TrainConfig(
        arch=args.arch,
        seed=args.seed,
        n_updates=args.updates,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        obs_noise=args.obs_noise,
        obs_perm=args.obs_perm,
        action_perm=args.action_perm,
        entropy_coef=args.entropy_coef,
        delta_coef=args.delta_coef,
        bptt=not args.no_bptt,
        lr=args.lr,
    )
    train(cfg)
