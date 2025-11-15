#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Inference-only script for running a pretrained policy on a robot.

Example:
```shell
python src/lerobot/scripts/lerobot_inference.py   \
--robot.type=so101_follower   \
--robot.port=/dev/so101_follower   \
--robot.id=follower   \
--robot.cameras='{ 
      camera1: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 25},
      camera2: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 25},
  }' \
--policy.pretrained_path=roboseasy/soarm_pick_and_place_blue_N_red_pens_smolvla   \
--policy.type=smolvla   \
--dataset.repo_id=roboseasy/soarm_pick_and_place_blue_red_merged   \
--instruction="Pick up the red pen and place it in the pencil case"   \
--display_data=true
```
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
# 코드에서 직접 사용하지 않지만, draccus parser가 카메라 타입을 레지스트리에 등록하기 위해 필요
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config, RobotConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, predict_action
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class DatasetConfig:
    repo_id: str


@dataclass
class InferenceConfig:
    robot: RobotConfig
    policy: PreTrainedConfig
    dataset: DatasetConfig
    display_data: bool = False
    instruction: str = ""
    fps: int = 25

    def __post_init__(self):
        # Load pretrained policy configuration if path is provided
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


def tokenize_instruction(policy, instruction: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize the instruction string for the VLA model.
    
    Args:
        policy: The pretrained policy (must have a tokenizer/processor)
        instruction: The natural language instruction string
        device: The device to place tensors on
        
    Returns:
        tuple: (language_tokens, language_attention_mask)
    """
    # Check if policy has a processor (for SmolVLA)
    if hasattr(policy, "model") and hasattr(policy.model, "vlm_with_expert"):
        processor = policy.model.vlm_with_expert.processor
        tokenizer = processor.tokenizer
        
        # Tokenize the instruction
        # Add special formatting if needed (e.g., "Question: {instruction} Answer:")
        formatted_instruction = f"Question: {instruction} Answer:"
        
        tokens = tokenizer(
            formatted_instruction,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        lang_tokens = tokens["input_ids"].to(device)
        lang_masks = tokens["attention_mask"].to(device)
        
        return lang_tokens, lang_masks
    else:
        # For policies that don't use language tokens, return empty tensors
        logging.warning("Policy does not support language tokens. Returning empty tensors.")
        return torch.zeros((1, 1), dtype=torch.long, device=device), torch.ones((1, 1), dtype=torch.bool, device=device)


@parser.wrap()
def main(cfg: InferenceConfig):
    init_logging()
    
    if cfg.display_data:
        init_rerun(session_name="inference")

    # Initialize robot
    robot = make_robot_from_config(cfg.robot)
    
    # Load dataset metadata to get features configuration
    ds_meta = LeRobotDataset(repo_id=str(cfg.dataset.repo_id)).meta
    
    # Load pretrained policy
    policy = make_policy(cfg.policy, ds_meta=ds_meta)
    
    # Create preprocessor and postprocessor pipelines with dataset stats for denormalization
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=ds_meta.stats,
    )
    
    # Create robot action and observation processors
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    device = get_safe_torch_device(policy.config.device)
    
    # Tokenize the instruction once at the beginning
    logging.info(f"Instruction: {cfg.instruction}")
    lang_tokens, lang_masks = tokenize_instruction(policy, cfg.instruction, device)
    
    # Reset policy and processors
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
    
    # Connect to robot
    robot.connect()
    
    # Initialize keyboard listener for graceful exit
    listener, events = init_keyboard_listener()
    
    try:
        logging.info("Starting inference loop. Press ESC to stop.")
        
        while not events.get("stop_recording", False):
            start_loop_t = time.perf_counter()
            
            if events.get("exit_early", False):
                logging.info("Early exit requested.")
                break
            
            # Get robot observation
            obs = robot.get_observation()
            
            # Apply observation processor pipeline (default is IdentityProcessor)
            obs_processed = robot_observation_processor(obs)
            
            # Build observation frame from processed observations
            observation_frame = build_dataset_frame(ds_meta.features, obs_processed, prefix=OBS_STR)
            
            # Add language tokens to observation
            observation_frame[OBS_LANGUAGE_TOKENS] = lang_tokens
            observation_frame[OBS_LANGUAGE_ATTENTION_MASK] = lang_masks
            
            # # DEBUG: Print observation frame keys and shapes
            # logging.info("=" * 80)
            # logging.info("DEBUG: observation_frame keys:")
            # for key in observation_frame.keys():
            #     if isinstance(observation_frame[key], torch.Tensor):
            #         logging.info(f"  {key}: {observation_frame[key].shape}")
            #     else:
            #         logging.info(f"  {key}: {type(observation_frame[key])} = {observation_frame[key]}")
            
            # Predict action using the policy
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=getattr(policy.config, "use_amp", False),
                task=cfg.instruction,
                robot_type=cfg.robot.type,
            )
            
            # DEBUG: Print action values
            logging.info("-" * 80)
            logging.info(f"DEBUG: action_values shape: {action_values.shape}")
            logging.info(f"DEBUG: action_values: {action_values}")
            logging.info("=" * 80)
            
            # Convert policy action to robot action format
            act_processed_policy = make_robot_action(action_values, ds_meta.features)
            logging.info(f"DEBUG: act_processed_policy: {act_processed_policy}")
            
            # Apply robot action processor pipeline
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))
            logging.info(f"DEBUG: robot_action_to_send: {robot_action_to_send}")
            
            # Send action to robot
            robot.send_action(robot_action_to_send)
            
            # Log data for visualization if enabled
            if cfg.display_data:
                # action_values는 Tensor이므로 딕셔너리로 변환한 act_processed_policy를 사용
                log_rerun_data(observation=obs_processed, action=act_processed_policy)
            
            # Maintain target FPS
            dt_s = time.perf_counter() - start_loop_t
            sleep_time = max(0, 1 / cfg.fps - dt_s)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        raise
    finally:
        # Cleanup
        logging.info("Disconnecting robot...")
        robot.disconnect()
        
        if listener is not None:
            listener.stop()
        
        logging.info("Inference completed.")


if __name__ == "__main__":
    main()
