"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import glob


def test_dyadic(trainer, img_path, segment_length):
    # Test using the first validation data
    test_data = next(trainer.val_dataloader)
    segment_name = test_data["segment"]
    start_idx = test_data["frame_idx"]
    suffix = str(start_idx)

    trainer.artifacts_dir = trainer.artifacts_dir + "/test/" + segment_name
    os.makedirs(trainer.artifacts_dir + "/viz", exist_ok=True)

    # Load participant images to visualize them side by side with the generated actor
    participant_imgs = load_participant_images(img_path, segment_name, segment_length)

    # Generate 3 samples
    # This will save a video concatenating in this order: [groundtruth, participant, generated 1, generated 2, generated 3]
    test_data = {k: v.unsqueeze(0) for k, v in test_data.items() if k not in ["frame_idx", "segment"]}
    trainer.test_step(test_data, suffix=suffix, images=participant_imgs, sample_rate=48000)


def load_participant_images(img_path, segment, segment_length, frame_idx=0, frame_rate=86.13281230):
    N_images = len(list(glob.glob(os.path.join(img_path, segment, "*.png"))))
    if segment in ["l--20240509--0942--GQS883--pilot--Chatsy", "l--20240509--1021--GQS883--pilot--Chatsy", "l--20240509--1052--GQS883--pilot--Chatsy"]:
        _step = 3
    else:
        _step = 1
    participant_images = [
        os.path.join(img_path, segment, f"{img_idx}.png")
        for img_idx in range(0, N_images, _step)
    ]

    participant_res = 30 / frame_rate
    participant_start = int(participant_res * frame_idx)
    participant_end = int(participant_res * (frame_idx + segment_length))
    if participant_end - participant_start > int(30 / frame_rate * segment_length):
        participant_end -= 1
    
    imgs = participant_images[participant_start:participant_end]

    return imgs
