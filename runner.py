import argparse
import json
import re
from os import remove
from pathlib import Path
import sys
import glob
import time
from datetime import datetime
import docker
from docker.types.containers import DeviceRequest


# 1. Parse arguments
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
optional = parser._action_groups.pop()
optional.add_argument('--runner-config', metavar="FILE",
                      default=Path(__file__).parent / 'cfg' / 'runner.json',
                      help='path to runner JSON configuration file')
optional.add_argument('--mot-config', metavar="FILE",
                      default=None,
                      help='path to mot JSON configuration file')
optional.add_argument('--root-folder', metavar="FILE",
                      default=Path(__file__).parent / 'video2',
                      help='path to root folder with video')
optional.add_argument('--file-mask', metavar="str",
                      default='*.mp4',
                      help='video file mask')
optional.add_argument('--containers', metavar="int",
                      default=None,
                      help='count of containers')
optional.add_argument('--threads', metavar="int",
                      default=None,
                      help='count of threads')

parser._action_groups.append(optional)
args = parser.parse_args()

with open(args.runner_config, 'r') as cfg_runer_file:
    runner_config = json.load(cfg_runer_file, cls=json.JSONDecoder,
                              object_hook=lambda d: type(sys.implementation)(**d))

if args.mot_config is not None:
    mot_config_file = open(Path(__file__).parent / args.mot_config, 'r')
else:
    mot_config_file = open(Path(__file__).parent /
                           runner_config.mot_config, 'r')
mot_config = json.load(mot_config_file, cls=json.JSONDecoder,
                       object_hook=lambda d: type(sys.implementation)(**d))

thread_count = args.threads
container_count = args.containers
if thread_count is None:
    thread_count = runner_config.threads
if container_count is None:
    container_count = runner_config.containers

# 2. Print parameters
print()
print("---- Hello from FastMOT runner ----")
print(F'{mot_config.mot_cfg.yolo_detector_cfg.model}')
print(F'{mot_config.stream_cfg.resolution[0]}x{mot_config.stream_cfg.resolution[1]} @ {mot_config.stream_cfg.frame_rate} fps')
print(F'Frame size: {mot_config.resize_to[0]}x{mot_config.resize_to[1]}')
print(F'Frame skip: {mot_config.mot_cfg.detector_frame_skip}')
print(F'Max age: {mot_config.mot_cfg.tracker_cfg.max_age}')

# 3. Prepare data
suffix = F'{mot_config.mot_cfg.yolo_detector_cfg.model}_{container_count}_{thread_count}_' \
    F'{mot_config.stream_cfg.resolution[0]}x{mot_config.stream_cfg.resolution[1]}_{mot_config.stream_cfg.frame_rate}_' \
    F'{mot_config.resize_to[0]}x{mot_config.resize_to[1]}_' \
    F'{mot_config.mot_cfg.detector_frame_skip}x{mot_config.mot_cfg.tracker_cfg.max_age}'

input_files = list(set(glob.iglob(f'{args.root_folder}/**/{args.file_mask}', recursive = True)) -
                   set(glob.iglob(f'{args.root_folder}/**/*fast_mot*', recursive = True)))

# Number of files should be divisible by thread_count * container_count
input_files = input_files[0: -(len(input_files) % (thread_count * container_count)) or None]

output_files = [str(Path(path).with_name(
    Path(path).stem + f'_fast_mot_{suffix}' + Path(path).suffix)) for path in input_files]
annotation_files = [str(Path(path).with_name(
    Path(path).stem + f'_fast_mot_{suffix}' + '.txt')) for path in input_files]

batched_input_uris = [' '.join(input_files[i:i + thread_count])
                      for i in range(0, len(input_files), thread_count)]
batched_output_uris = [' '.join(output_files[i:i + thread_count])
                       for i in range(0, len(output_files), thread_count)]
batched_annotation_uris = [' '.join(annotation_files[i:i + thread_count])
                           for i in range(0, len(annotation_files), thread_count)]

print()
print(f'Root folder: {args.root_folder}')
print(f'File mask: {args.file_mask}')
print(f'Files: {len(input_files)}')
print(f'Batches: {len(batched_input_uris)}')
print()

# 4. Run docker containers
print('Start processing...')
print(f'{container_count} containers - {thread_count} threads')

t0 = time.time()
log_file_name = str(Path(__file__).parent) + datetime.utcnow().strftime(f'/%Y%m%d_%H%M%S_{suffix}')[:-3] + '.txt'

total = 0
count = 0
client = docker.from_env()
for batch_idx in range(0, len(batched_input_uris)):
    input_uri = batched_input_uris[batch_idx]
    output_uri = batched_output_uris[batch_idx]
    annotation_uri = batched_annotation_uris[batch_idx]

    # --output-uri {output_uri} 
    cmd_line = f'python app.py --input-uri {input_uri} --txt {annotation_uri} --mot --verbose'
    container = client.containers.run('fastmot:latest', 
                                      cmd_line,
                                      detach = True,
                                      network_mode='host',
                                      volumes = {Path(__file__).parent.resolve(): {'bind': '/usr/src/app/FastMOT', 'mode': 'rw'},
                                                 '/tmp/.X11-unix': {'bind': '/tmp/.X11-unix', 'mode': 'rw'}},
                                      device_requests = [DeviceRequest(driver = 'nvidia', count = -1, capabilities = [['compute', 'utility', 'video']])]
                                     )
    print(f'{container}')

    while len(client.containers.list(filters = {'status': 'running'})) >= container_count:
        time.sleep(0.1)
    
    for container in client.containers.list(filters = {'status': 'exited'}):
        with open(log_file_name, 'a') as log_file:
            log_str = str(container.logs())
            log_str = log_str.replace('\\n', '\n')
            log_str = log_str.replace('\\r', '\r')
            effective_fps_line = str([s for s in log_str.splitlines() if 'Effective FPS' in s])
            matches = list(re.finditer('\  ([0-9]+)', effective_fps_line))
            total += sum([float(m.group(0)) for m in matches])
            count += len(matches)
            log_file.write(log_str)
        container.remove()

for container in client.containers.list(all=True):
    container.wait()
    with open(log_file_name, 'a') as log_file:
        log_str = str(container.logs())
        log_str = log_str.replace('\\n', '\n')
        log_str = log_str.replace('\\r', '\r')
        effective_fps_line = str([s for s in log_str.splitlines() if 'Effective FPS' in s])
        matches = list(re.finditer('\  ([0-9]+)', effective_fps_line))
        total += sum([float(m.group(0)) for m in matches])
        count += len(matches)
        log_file.write(log_str)
    container.remove()

print()
print(f'{suffix} - {int(time.time() - t0)}s - effective FPS: {int(total / count)}')
print()
