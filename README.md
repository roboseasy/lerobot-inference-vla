
lerobot_inference.py 이 코드 파일을 아래와 같은 위치에 놓아주세요.

`src/lerobot/scripts/lerobot_inference.py`




아래와 같이 입력하면, lerobot-record.py 로 추론할때 폴더가 저장되는 귀찮음에서 벗어날 수 있습니다.

```shell
python src/lerobot/scripts/lerobot_inference.py   \
	--robot.type=so101_follower \
	--robot.port=/dev/so101_follower \
	--robot.id=follower \
	--robot.cameras='{
camera1: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 25},
camera2: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 25},
}' \
  --policy.pretrained_path=roboseasy/soarm_pick_and_place_blue_N_red_pens_smolvla \
  --policy.type=smolvla \
  --dataset.repo_id=roboseasy/soarm_pick_and_place_blue_red_merged \
  --instruction="Pick up the red pen and place it in the pencil case" \
  --display_data=true
```

또는 

`~/lerobot/pyproject.toml` 해당 위치의 파일에

```toml
[project.scripts]
```

에 아래 내용을 추가하고

```toml
lerobot-inference="lerobot.scripts.lerobot_inference:main"
```

아래 명령어를 통해 빌드해줍니다.

```bash

pip install -e .

```

그러면 이제 아래와 같은 명령어로 훈련한 정책을 추론할 수 있습니다.

```bash

lerobot-inference \
	--robot.type=so101_follower \
	--robot.port=/dev/so101_follower \
	--robot.id=follower \
	--robot.cameras='{
		front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 25},
		side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 25},
	    }' \
	--policy.path=${HF_USER}/act_${TASK_NAME} \
	--instruction="${TASK_DESCRIPTION}" \
	--display_data=true

```