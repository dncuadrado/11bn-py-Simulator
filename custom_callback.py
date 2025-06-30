import os
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization, VecEnv
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback # Import from sb3_contrib

class CustomMaskableEvalCallback(MaskableEvalCallback):
    """
    Custom EvalCallback that signals the evaluation context to the eval_env.
    Inherits from MaskableEvalCallback to retain masking support.

    Sets a temporary attribute `_is_evaluating` on the eval_env during evaluation.
    """
    def __init__(self, *args, use_masking: bool = True, **kwargs):
        # Initialize the parent class (MaskableEvalCallback)
        super().__init__(*args, use_masking=use_masking, **kwargs)
        # Note: use_masking is already handled by MaskableEvalCallback's init

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # --- Signal Start of Evaluation ---
            # Set the evaluation flag before evaluation begins
            try:
                if isinstance(self.eval_env, VecEnv):
                    # Use set_attr for VecEnv environments
                    self.eval_env.set_attr("_is_evaluating", True)
                else:
                    # Use standard setattr for single environments
                    setattr(self.eval_env, "_is_evaluating", True)

                # --- Core Evaluation Logic (Adapted from MaskableEvalCallback._on_step) ---
                # Sync training and eval env if there is VecNormalize
                if self.model.get_vec_normalize_env() is not None:
                    try:
                        sync_envs_normalization(self.training_env, self.eval_env)
                    except AttributeError as e:
                        raise AssertionError(
                            "Training and eval env are not wrapped the same way, "
                            "see stable-baselines3 docs on EvalCallback."
                        ) from e

                # Reset success rate buffer
                self._is_success_buffer = []

                # Call evaluate_policy (uses the patched version for masking)
                episode_rewards, episode_lengths = evaluate_policy(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    render=self.render,
                    deterministic=self.deterministic,
                    return_episode_rewards=True,
                    warn=self.warn,
                    callback=self._log_success_callback,
                    use_masking=self.use_masking, # Pass masking flag
                )

                # --- Logging and Saving ---
                if self.log_path is not None:
                    assert isinstance(episode_rewards, list)
                    assert isinstance(episode_lengths, list)
                    self.evaluations_timesteps.append(self.num_timesteps)
                    self.evaluations_results.append(episode_rewards)
                    self.evaluations_length.append(episode_lengths)

                    kwargs = {}
                    # Save success log if present
                    if len(self._is_success_buffer) > 0:
                        self.evaluations_successes.append(self._is_success_buffer)
                        kwargs = dict(successes=self.evaluations_successes)

                    np.savez(
                        self.log_path,
                        timesteps=self.evaluations_timesteps,
                        results=self.evaluations_results,
                        ep_lengths=self.evaluations_length,
                        **kwargs,
                    )

                mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
                mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
                self.last_mean_reward = float(mean_reward)

                if self.verbose > 0:
                    print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.6f} +/- {std_reward:.6f}")
                    print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

                self.logger.record("eval/mean_reward", float(mean_reward))
                self.logger.record("eval/mean_ep_length", mean_ep_length)

                if len(self._is_success_buffer) > 0:
                    success_rate = np.mean(self._is_success_buffer)
                    if self.verbose > 0:
                        print(f"Success rate: {100 * success_rate:.2f}%")
                    self.logger.record("eval/success_rate", success_rate)

                # Dump log so the evaluation results are printed with the correct timestep
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(self.num_timesteps)

                # Check for new best model and save if necessary
                if mean_reward > self.best_mean_reward:
                    if self.verbose > 0:
                        print("New best mean reward!")
                    if self.best_model_save_path is not None:
                        self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    self.best_mean_reward = float(mean_reward)
                    # Trigger callback on new best model, if needed
                    if self.callback_on_new_best is not None:
                        continue_training = self.callback_on_new_best.on_step()

                # Trigger callback after every evaluation, if needed
                if self.callback is not None:
                    continue_training = continue_training and self._on_event()

            finally:
                # --- Signal End of Evaluation ---
                # Ensure the flag is reset even if an error occurs during evaluation
                if isinstance(self.eval_env, VecEnv):
                    self.eval_env.set_attr("_is_evaluating", False)
                else:
                    setattr(self.eval_env, "_is_evaluating", False)

        return continue_training