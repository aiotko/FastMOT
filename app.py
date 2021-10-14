#!/usr/bin/env python3

from pathlib import Path
from types import SimpleNamespace
import argparse
import logging
import json
import cv2
import threading

import fastmot
import fastmot.models
from fastmot.utils import ConfigDecoder, Profiler

def do_magic(config, stream, stream_num, mot, output_uri, output_rtsp, txt, show, video_window_name, logger, profiler):
    try:
        while not show or cv2.getWindowProperty(video_window_name, 0) >= 0:
            frame = stream.read()
            if frame is None:
                break

            if mot is not None:
                mot.step(frame)
                if txt is not None:
                    for track in mot.visible_tracks():
                        tl = track.tlbr[:2] / config.resize_to * stream.resolution
                        br = track.tlbr[2:] / config.resize_to * stream.resolution
                        w, h = br - tl + 1
                        txt.write(f'{mot.frame_count},{track.trk_id},{tl[0]:.6f},{tl[1]:.6f},'
                                f'{w:.6f},{h:.6f},{track.conf:.6f},-1,-1,-1\n')

            if show:
                cv2.imshow(video_window_name, frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if output_uri is not None:
                stream.write(frame)

            if output_rtsp is not None:
                stream.write_rtsp(frame)
    finally:
        if txt is not None:
            txt.close()
        stream.release()

    # timing statistics
    if mot is not None:
        avg_fps = round(mot.frame_count / profiler.duration)
        logger.info(f'Average FPS (stream #{stream_num}): %d', avg_fps)
        mot.print_timing_info()        

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    group = parser.add_mutually_exclusive_group()
    required.add_argument('-i', '--input-uri', metavar="URI", nargs='+', type=str, required=True, help=
                          'URI to input stream\n'
                          '1) image sequence (e.g. %%06d.jpg)\n'
                          '2) video file (e.g. file.mp4)\n'
                          '3) MIPI CSI camera (e.g. csi://0)\n'
                          '4) USB camera (e.g. /dev/video0)\n'
                          '5) RTSP stream (e.g. rtsp://<user>:<password>@<ip>:<port>/<path>)\n'
                          '6) HTTP stream (e.g. http://<user>:<password>@<ip>:<port>/<path>)\n')
    optional.add_argument('-c', '--config', metavar="FILE",
                          default=Path(__file__).parent / 'cfg' / 'mot.json',
                          help='path to JSON configuration file')
    optional.add_argument('-l', '--labels', metavar="FILE",
                          help='path to label names (e.g. coco.names)')
    optional.add_argument('-o', '--output-uri', metavar="URI", nargs='+', type=str,
                          help='URI to output video file')
    optional.add_argument('-r', '--output-rtsp', metavar="URI", nargs='+', type=str,
                          help='RTSP Server URI to output video')
    optional.add_argument('-t', '--txt', metavar="FILE",
                          help='path to output MOT Challenge format results (e.g. MOT20-01.txt)')
    optional.add_argument('-m', '--mot', action='store_true', help='run multiple object tracker')
    optional.add_argument('-s', '--show', action='store_true', help='show visualizations')
    group.add_argument('-q', '--quiet', action='store_true', help='reduce output verbosity')
    group.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
    parser._action_groups.append(optional)
    args = parser.parse_args()
    if args.txt is not None and not args.mot:
        raise parser.error('argument -t/--txt: not allowed without argument -m/--mot')

    # set up logging
    logging.basicConfig(format='%(asctime)s [%(levelname)8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(fastmot.__name__)
    if args.quiet:
        logger.setLevel(logging.WARNING)
    elif args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # load config file
    with open(args.config) as cfg_file:
        config = json.load(cfg_file, cls=ConfigDecoder, object_hook=lambda d: SimpleNamespace(**d))

    # load labels if given
    if args.labels is not None:
        with open(args.labels) as label_file:
            label_map = label_file.read().splitlines()
            fastmot.models.set_label_map(label_map)

    streams = []
    mots = [] if args.mot else None
    txts = [] if args.txt else None
    mot = None
    txt = None
    video_window_name = None
    draw = args.show or args.output_uri is not None or args.output_rtsp is not None    

    try:
        with Profiler('app') as prof:
            stream_count = len(args.input_uri)
            for stream_num in range(0, stream_count):
                output_rtsp = args.output_rtsp[stream_num] if args.output_rtsp is not None else None
                output_uri = args.output_uri[stream_num] if args.output_uri is not None else None
                streams.append(fastmot.VideoIO(config.resize_to, args.input_uri[stream_num], output_uri, output_rtsp, **vars(config.stream_cfg)))
                if args.mot:
                    mot = fastmot.MOT(config.resize_to, **vars(config.mot_cfg), draw=draw)
                    mots.append(mot)
                    mots[stream_num].reset(streams[stream_num].cap_dt)
                
                if args.txt is not None:
                    Path(args.txt[stream_num]).parent.mkdir(parents=True, exist_ok=True)
                    txt = open(args.txt[stream_num], 'w')
                    txts.append(txt)

                if args.show:
                    video_window_name = f'Video {stream_num}'
                    cv2.namedWindow(video_window_name, cv2.WINDOW_AUTOSIZE)

                logger.info('Starting video capture...')
                streams[stream_num].start_capture()

                t = threading.Thread(target=do_magic, args=(config, streams[stream_num], stream_num, mot, output_uri, output_rtsp, txt, args.show, video_window_name, logger, prof,))
                t.start()
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()