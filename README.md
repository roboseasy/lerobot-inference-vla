
lerobot_inference.py 이 코드 파일을 아래와 같은 위치에 놓아주세요.

src/lerobot/scripts/lerobot_inference.py


아래와 같이 입력하면 이제, lerobot-record.py 로 추론할때 폴더가 저장되는 귀찮음에서 벗어날 수 있습니다.

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
