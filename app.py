#!/usr/bin/env python3

from pathlib import Path
from types import SimpleNamespace
import argparse
import logging
import time
import json
import cv2
import threading

import fastmot
import fastmot.models
from fastmot.utils import ConfigDecoder, Profiler

frame_count = {}

def do_magic(config, stream, stream_num, mot, output_uri, output_rtsp, txt, show, video_window_name, logger, profiler):
    with profiler:
        try:
            with Profiler(stream_num, 'effective'):
                count = 0
                t = time.time()
                while not show or cv2.getWindowProperty(video_window_name, 0) >= 0:
                    with Profiler(stream_num, 'read'):
                        frame = stream.read()
                        #if count == 1000:
                        #    break
                        if frame is None:
                            break
                        count += 1
                        if count % 100 == 0:
                            logger.debug(f"FPS: {100 / (time.time() - t)}")
                            t = time.time()

                    if mot is not None:
                        with Profiler(stream_num, 'mot'):
                            mot.step(frame)
                        with Profiler(stream_num, 'txt'):
                            if txt is not None:
                                for track in mot.visible_tracks():
                                    tl = track.tlbr[:2] / config.resize_to * stream.resolution
                                    br = track.tlbr[2:] / config.resize_to * stream.resolution
                                    w, h = br - tl + 1
                                    txt.write(f'{mot.frame_count},{track.trk_id},{tl[0]:.6f},{tl[1]:.6f},'
                                            f'{w:.6f},{h:.6f},{track.conf:.6f},-1,-1,-1\n')

                    if show:
                        with Profiler(stream_num, 'show'):
                            cv2.imshow(video_window_name, frame)
                            if cv2.waitKey(1) & 0xFF == 27:
                                break

                    if output_uri is not None:
                        with Profiler(stream_num, 'write'):
                            stream.write(frame)

                    if output_rtsp is not None:
                        with Profiler(stream_num, 'rtsp'):
                            stream.write_rtsp(frame)
                frame_count[stream_num] = count
        finally:
            if txt is not None:
                txt.close()
            stream.release() 


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
    optional.add_argument('-t', '--txt', metavar="FILE", nargs='+', type=str,
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
        threads = []
        stream_count = len(args.input_uri)
        for stream_num in range(0, stream_count):
            with Profiler(stream_num, 'app') as prof:
                output_rtsp = args.output_rtsp[stream_num] if args.output_rtsp is not None else None
                output_uri = args.output_uri[stream_num] if args.output_uri is not None else None
                streams.append(fastmot.VideoIO(config.resize_to, args.input_uri[stream_num], output_uri, output_rtsp, **vars(config.stream_cfg)))
                if args.mot:
                    mot = fastmot.MOT(stream_num, config.resize_to, **vars(config.mot_cfg), draw=draw)
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

                threads.append(threading.Thread(target=do_magic, args=(config, streams[stream_num], stream_num, mot, output_uri, output_rtsp, txt, args.show, video_window_name, logger, prof,)))
        
        for stream_num in range(0, stream_count):
            threads[stream_num].start()

        for stream_num in range(0, stream_count):
            threads[stream_num].join()

        avg_fps_msg = f"{'Average FPS: ':<39}"
        eff_fps_msg = f"{'Effective FPS: ':<39}"
        tot_time_msg = f"{'Total time:':<39}"
        eff_time_msg = f"{'Effective time:':<39}"
        frm_cnt_msg = f"{'Frames count:':<39}"
        eff_time_per_frm_msg = f"{'Effective time:':<39}"
        read_frm_msg = f"{'Read frames:':<39}"
        wrt_ann_msg = f"{'Write annonations':<39}"
        show_msg = f"{'Show:':<39}"
        wrt_frm_msg = f"{'Write frames':<39}"
        out_rtsp_msg = f"{'Output RTSP:':<39}"
        mot_msg = f"{'MOT:':<39}"
        init_msg = f"{'  init time:':<39}"
        prep_msg = f"{'  preprocess time:':<39}"
        det_prep_msg = f"{'    preprocess time:':<39}"
        det_infer_async_msg = f"{'    infer async time:':<39}"
        track_msg =f"{'  track time:':<39}"
        f1_msg = f"{'    f1:':<39}"
        f2_msg = f"{'    f2:':<39}"
        f3_msg = f"{'    f3:':<39}"
        f4_msg = f"{'    f4:':<39}"
        f5_msg = f"{'    f5:':<39}"
        f6_msg = f"{'    f6:':<39}"
        f7_msg = f"{'    f7:':<39}"
        f8_msg = f"{'    f8:':<39}"
        f9_msg = f"{'    f9:':<39}"
        f9x_msg = f"{'     cv2.calcOpticalFlowPyrLK:':<39}"
        f10_msg = f"{'    f10:':<39}"
        f11_msg = f"{'    f11:':<39}"
        f12_msg = f"{'    f12:':<39}"
        f13_msg = f"{'    f13:':<39}"
        f14_msg = f"{'    f14:':<39}"
        f15_msg = f"{'    f15:':<39}"
        det_msg = f"{'  detect time:':<39}"
        extr1_msg = f"{'  feature extract time (phase 1):':<39}"
        kalman_msg = f"{'  kalman filter time:':<39}"
        extr2_msg = f"{'  feature extract time (phase 2):':<39}"
        ass_msg = f"{'  association time:':<39}"

        for stream_num in range(0, stream_count):
            avg_fps = round(frame_count[stream_num] * 1000 / Profiler.get_avg_millis(stream_num, 'app'))
            effective_fps = round(frame_count[stream_num] * 1000 / Profiler.get_avg_millis(stream_num, 'effective'))

            avg_fps_msg += f"{avg_fps:>8d}  "
            eff_fps_msg += f"{effective_fps:>8d}  "
            tot_time_msg += f"{Profiler.get_millis(stream_num, 'app') / 1000 :>8.3f}  "
            eff_time_msg += f"{Profiler.get_millis(stream_num, 'effective') / 1000 :>8.3f}  "
            frm_cnt_msg += f"{frame_count[stream_num]:>8d}  "
            eff_time_per_frm_msg += f"{Profiler.get_millis(stream_num, 'effective') / frame_count[stream_num]:>8.3f}  "
            read_frm_msg += f"{Profiler.get_millis(stream_num, 'read'):>8.3f}  "
            wrt_ann_msg += f"{Profiler.get_millis(stream_num, 'txt'):>8.3f}  "
            show_msg += f"{Profiler.get_millis(stream_num, 'show'):>8.3f}  "
            wrt_frm_msg += f"{Profiler.get_millis(stream_num, 'write'):>8.3f}  "
            out_rtsp_msg += f"{Profiler.get_millis(stream_num, 'rtsp'):>8.3f}  "
            mot_msg += f"{Profiler.get_millis(stream_num, 'mot'):>8.3f}  "

            init_msg += f"{Profiler.get_millis(stream_num, 'init'):>8.3f}  "
            prep_msg += f"{Profiler.get_millis(stream_num, 'preproc'):>8.3f}  "
            det_prep_msg += f"{Profiler.get_millis(stream_num, 'detect_preproc'):>8.3f}  "
            det_infer_async_msg += f"{Profiler.get_millis(stream_num, 'detect_infer_async'):>8.3f}  "
            track_msg += f"{Profiler.get_millis(stream_num, 'track'):>8.3f}  "
            f1_msg += f"{Profiler.get_millis(stream_num, 'f1'):>8.3f}  "
            f2_msg += f"{Profiler.get_millis(stream_num, 'f2'):>8.3f}  "
            f3_msg += f"{Profiler.get_millis(stream_num, 'f3'):>8.3f}  "
            f4_msg += f"{Profiler.get_millis(stream_num, 'f4'):>8.3f}  "
            f5_msg += f"{Profiler.get_millis(stream_num, 'f5'):>8.3f}  "
            f6_msg += f"{Profiler.get_millis(stream_num, 'f6'):>8.3f}  "
            f7_msg += f"{Profiler.get_millis(stream_num, 'f7'):>8.3f}  "
            f8_msg += f"{Profiler.get_millis(stream_num, 'f8'):>8.3f}  "
            f9_msg += f"{Profiler.get_millis(stream_num, 'f9'):>8.3f}  "
            f9x_msg += f"{Profiler.get_millis(stream_num, 'f9x'):>8.3f}  "
            f10_msg += f"{Profiler.get_millis(stream_num, 'f10'):>8.3f}  "
            f11_msg += f"{Profiler.get_millis(stream_num, 'f11'):>8.3f}  "
            f12_msg += f"{Profiler.get_millis(stream_num, 'f12'):>8.3f}  "
            f13_msg += f"{Profiler.get_millis(stream_num, 'f13'):>8.3f}  "
            f14_msg += f"{Profiler.get_millis(stream_num, 'f14'):>8.3f}  "
            f15_msg += f"{Profiler.get_millis(stream_num, 'f15'):>8.3f}  "

            det_msg += f"{Profiler.get_millis(stream_num, 'detect'):>8.3f}  "
            extr1_msg += f"{Profiler.get_millis(stream_num, 'extract1'):>8.3f}  "
            kalman_msg += f"{Profiler.get_millis(stream_num, 'kalman'):>8.3f}  "
            extr2_msg += f"{Profiler.get_millis(stream_num, 'extract2'):>8.3f}  "
            ass_msg += f"{Profiler.get_millis(stream_num, 'assoc'):>8.3f}  "

        logger.info(5 * stream_count * '=' + '============= Timing Stats ==============' + 5 * stream_count * '=')
        logger.info(avg_fps_msg)
        logger.debug(eff_fps_msg)
        logger.debug(frm_cnt_msg)
        logger.info(tot_time_msg + "s")
        logger.debug(eff_time_msg + "s")
        logger.debug(5 * stream_count * '=' + '=============== Per frame ===============' + 5 * stream_count * '=')
        logger.debug(read_frm_msg + "ms")
        logger.debug(wrt_ann_msg + "ms")
        logger.debug(show_msg + "ms")
        logger.debug(wrt_frm_msg + "ms")
        logger.debug(out_rtsp_msg + "ms")
        logger.debug(mot_msg + "ms")
        logger.debug(init_msg + "ms")
        logger.debug(prep_msg + "ms")
        logger.debug(det_prep_msg + "ms")
        logger.debug(det_infer_async_msg + "ms")
        logger.debug(track_msg + "ms")
        logger.debug(f1_msg +"ms")
        logger.debug(f2_msg +"ms")
        logger.debug(f3_msg +"ms")
        logger.debug(f4_msg +"ms")
        logger.debug(f5_msg +"ms")
        logger.debug(f6_msg +"ms")
        logger.debug(f7_msg +"ms")
        logger.debug(f8_msg +"ms")
        logger.debug(f9_msg +"ms")
        logger.debug(f9x_msg +"ms")
        logger.debug(f10_msg +"ms")
        logger.debug(f11_msg +"ms")
        logger.debug(f12_msg +"ms")
        logger.debug(f13_msg +"ms")
        logger.debug(f14_msg +"ms")
        logger.debug(f15_msg +"ms")
        logger.debug(det_msg + "ms")
        logger.debug(extr1_msg + "ms")
        logger.debug(kalman_msg + "ms")
        logger.debug(extr2_msg + "ms")
        logger.debug(ass_msg + "ms")
        
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
