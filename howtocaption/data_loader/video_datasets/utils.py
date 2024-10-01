import numpy as np
import torch as th
import ffmpeg


def get_video_frames(video_path, start, end, num_frames, fps=None, width=None, height=None,
                     sample_beginning=False, central_frames=False):
    try:
        if (width is None) or (height is None) or (end is None):
            probe = ffmpeg.probe(video_path)

        if end is None:
            end = float(probe['format']['duration'])
            if end == 0:
                end = num_frames
    except Exception as excep:
        print("Warning: ffmpeg error. video path: {} error. Error: {}".format(video_path, excep), flush=True)

    num_sec = end - start
    if fps is None:
        fps = num_frames / num_sec

        assert (sample_beginning == False) or (central_frames == False)
        if sample_beginning:
            start = start + np.random.random() * (num_sec / num_frames)
        if central_frames:
            start = start + (num_sec / num_frames) / 2

    cmd = (
        ffmpeg
            .input(video_path, ss=start, t=num_sec + 0.1)
            .filter('fps', fps=fps)
    )
    for i in range(1):
        try:
            if width is None:
                width = int(probe['streams'][0]['width'])

            if height is None:
                height = int(probe['streams'][0]['height'])

            out, _ = (
                cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                    .run(capture_stdout=True, quiet=True)
            )

            video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
            video = th.tensor(video)
            video = video.permute(0, 3, 1, 2)
            if video.shape[0] < num_frames:
                # print(f'Warning: sampling less frames than necessary:  {video.shape[0]}')
                zeros = th.zeros((num_frames - video.shape[0], 3, height, width), dtype=th.uint8)
                video = th.cat((video, zeros), axis=0)
            elif video.shape[0] > num_frames:
                # print(f'Warning: sampling more frames than necessary:  {video.shape[0]}')
                video = video[:num_frames]
            break
        except Exception as excep:
            print("Warning: ffmpeg error. video path: {} error. Error: {}".format(video_path, excep), flush=True)
    else:
        # print(f'Warning: ffmpeg error. {video_path}', flush=True)
        video = th.zeros((num_frames, 3, 224, 224), dtype=th.uint8)

    video = video.float() / 255.

    return video