import multiprocessing
import threading
from types import SimpleNamespace
from enum import Enum
import logging
import numpy as np
import cv2

from .detector import SSDDetector, YOLODetector, PublicDetector
from .feature_extractor import FeatureExtractor
from .tracker import MultiTracker
from .utils import Profiler
from .utils.visualization import Visualizer
from .utils.numba import find_split_indices

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process


LOGGER = logging.getLogger(__name__)


class DetectorType(Enum):
    SSD = 0
    YOLO = 1
    PUBLIC = 2


def my_track(arg):
    arg[0].track(arg[1], arg[2])
class MOT:
    def __init__(self, 
                 stream_num, 
                 detector,
                 size,
                 detector_type='YOLO',
                 detector_frame_skip=5,
                 class_ids=(1,),
                 ssd_detector_cfg=None,
                 yolo_detector_cfg=None,
                 public_detector_cfg=None,
                 feature_extractor_cfgs=None,
                 tracker_cfg=None,
                 visualizer_cfg=None,
                 draw=False):
        """Top level module that integrates detection, feature extraction,
        and tracking together.

        Parameters
        ----------
        size : tuple
            Width and height of each frame.
        detector_type : {'SSD', 'YOLO', 'public'}, optional
            Type of detector to use.
        detector_frame_skip : int, optional
            Number of frames to skip for the detector.
        class_ids : sequence, optional
            Class IDs to track. Note class ID starts at zero.
        ssd_detector_cfg : SimpleNamespace, optional
            SSD detector configuration.
        yolo_detector_cfg : SimpleNamespace, optional
            YOLO detector configuration.
        public_detector_cfg : SimpleNamespace, optional
            Public detector configuration.
        feature_extractor_cfgs : List[SimpleNamespace], optional
            Feature extractor configurations for all classes.
            Each configuration corresponds to the class at the same index in sorted `class_ids`.
        tracker_cfg : SimpleNamespace, optional
            Tracker configuration.
        visualizer_cfg : SimpleNamespace, optional
            Visualization configuration.
        draw : bool, optional
            Draw visualizations.
        """
        self.stream_num = stream_num
        self.size = size
        self.detector_type = DetectorType[detector_type.upper()]
        assert detector_frame_skip >= 1
        self.detector_frame_skip = detector_frame_skip
        self.class_ids = tuple(np.unique(class_ids))
        self.draw = draw

        if ssd_detector_cfg is None:
            ssd_detector_cfg = SimpleNamespace()
        if yolo_detector_cfg is None:
            yolo_detector_cfg = SimpleNamespace()
        if public_detector_cfg is None:
            public_detector_cfg = SimpleNamespace()
        if feature_extractor_cfgs is None:
            feature_extractor_cfgs = (SimpleNamespace(),)
        if tracker_cfg is None:
            tracker_cfg = SimpleNamespace()
        if visualizer_cfg is None:
            visualizer_cfg = SimpleNamespace()
        if len(feature_extractor_cfgs) != len(class_ids):
            raise ValueError('Number of feature extractors must match length of class IDs')

        # LOGGER.info('Loading detector model...')
        # if self.detector_type == DetectorType.SSD:
        #     self.detector = SSDDetector(stream_num, self.size, self.class_ids, **vars(ssd_detector_cfg))
        # elif self.detector_type == DetectorType.YOLO:
        #     self.detector = YOLODetector(stream_num, self.size, self.class_ids, **vars(yolo_detector_cfg))
        # elif self.detector_type == DetectorType.PUBLIC:
        #     self.detector = PublicDetector(stream_num, self.size, self.class_ids, self.detector_frame_skip,
        #                                    **vars(public_detector_cfg))
        # TODO
        self.detector = detector

        LOGGER.info('Loading feature extractor models...')
        self.extractors = [[FeatureExtractor(**vars(cfg))for cfg in feature_extractor_cfgs]  for _ in range(0, self.stream_num) ]
        self.trackers = [MultiTracker(self.size, self.extractors[stream_idx][0].metric, **vars(tracker_cfg)) for stream_idx in range(0, self.stream_num)]
        self.visualizer = Visualizer(**vars(visualizer_cfg))
        self.frame_count = 0

    def visible_tracks(self):
        """Retrieve visible tracks from the tracker

        Returns
        -------
        Iterator[Track]
            Confirmed and active tracks from the tracker.
        """
        return [(track for track in self.trackers[stream_idx].tracks.values()
                if track.confirmed and track.active) for stream_idx in range(0, self.stream_num)]

    def reset(self, cap_dt):
        """Resets multiple object tracker. Must be called before `step`.

        Parameters
        ----------
        cap_dt : float
            Time interval in seconds between each frame.
        """
        self.frame_count = 0
        for stream_idx in range(0, self.stream_num):
            self.trackers[stream_idx].reset(cap_dt)

    def step(self, frames, stream):
        """Runs multiple object tracker on the next frame.

        Parameters
        ----------
        frame : ndarray
            The next frame.
        """
        detections = [None] * self.stream_num
        if self.frame_count == 0:
            with Profiler(0, 'init'):
                detections = self.detector(frames)
                for stream_idx in range(0, self.stream_num):
                    self.trackers[stream_idx].init(frames[stream_idx], detections[stream_idx])
            with Profiler(0, 'read'):
                next_frames = stream.read()
        elif self.frame_count % self.detector_frame_skip == 0:
            with Profiler(0, 'preproc'):
                self.detector.detect_async(frames, True)

            threads = []
            with Profiler(0, 'track'):
                for stream_idx in range(0, self.stream_num):
                    threads.append(threading.Thread(target=self.trackers[stream_idx].compute_flow, args=(stream_idx, frames[stream_idx], )))
                    threads[stream_idx].start()
            # with Profiler(0, 'track'):
            #     pool = ThreadPoolExecutor(max_workers=21)
            #     for stream_idx in range(0, self.stream_num):
            #         pool.submit(self.trackers[stream_idx].compute_flow, stream_idx, frames[stream_idx])
            # processes = []
            # with Profiler(0, 'track'):
            #     for stream_idx in range(0, self.stream_num):
            #         ctx = multiprocessing.get_context('forkserver')
            #         p = ctx.Process(target=self.trackers[stream_idx].compute_flow, args=(stream_idx, frames[stream_idx], ))
            #         p.start()
            #         processes.append(p)

            with Profiler(0, 'read'):
                next_frames = stream.read()
            
            #with Profiler(0, 'track'):
            for stream_idx in range(0, self.stream_num):
                threads[stream_idx].join()
            # pool.shutdown(wait=True)
            # for stream_idx in range(0, self.stream_num):
            #     processes[stream_idx].join()

            with Profiler(0, 'detect'):
                detections = self.detector.postprocess(True)

            for stream_idx in range(0, self.stream_num):
                with Profiler(stream_idx, 'extract1'):
                    cls_bboxes = np.split(detections[stream_idx].tlbr, find_split_indices(detections[stream_idx].label))
                    for extractor, bboxes in zip(self.extractors[stream_idx], cls_bboxes):
                        extractor.extract_async(frames[stream_idx], bboxes)

            for stream_idx in range(0, self.stream_num):
                self.trackers[stream_idx].apply_kalman(stream_idx)

            embeddings = [[]] * self.stream_num
            for stream_idx in range(0, self.stream_num):
                with Profiler(stream_idx, 'extract2'):
                    for extractor in self.extractors[stream_idx]:
                        embeddings[stream_idx].append(extractor.postprocess())
                    embeddings[stream_idx] = np.concatenate(embeddings[stream_idx]) if len(embeddings[stream_idx]) > 1 else embeddings[stream_idx][0]

            for stream_idx in range(0, self.stream_num):
                with Profiler(stream_idx, 'assoc'):
                    self.trackers[stream_idx].update(self.frame_count, detections[stream_idx], embeddings[stream_idx])
        else:
            threads = []
            with Profiler(0, 'track'):
                for stream_idx in range(0, self.stream_num):
                    threads.append(threading.Thread(target=self.trackers[stream_idx].track, args=(stream_idx, frames[stream_idx], )))
                    threads[stream_idx].start()
            # with Profiler(0, 'track'):
            #     pool = ThreadPoolExecutor(max_workers=21)
            #     for stream_idx in range(0, self.stream_num):
            #         pool.submit(self.trackers[stream_idx].track, stream_idx, frames[stream_idx])

            # processes = []
            # with Profiler(0, 'track'):
            #     for stream_idx in range(0, self.stream_num):
            #         ctx = multiprocessing.get_context('forkserver')
            #         p = ctx.Process(target=self.trackers[stream_idx].track, args=(stream_idx, frames[stream_idx], ))
            #         p.start()
            #         processes.append(p)

            # with Profiler(0, 'track'):
            #     arg = []
            #     for stream_idx in range(0, self.stream_num):
            #         arg.append([self.trackers[stream_idx].track, stream_idx, frames[stream_idx]])
            #     pool = multiprocessing.Pool(processes=self.stream_num)     
            #     pool.map(my_track, arg)           
            
            with Profiler(0, 'read'):
                next_frames = stream.read()

            #with Profiler(0, 'track'):
            for stream_idx in range(0, self.stream_num):
                threads[stream_idx].join()
            # pool.shutdown(wait=True)
            # for stream_idx in range(0, self.stream_num):
            #     processes[stream_idx].join()
            # pool.close()
            # pool.join()

        if self.draw:
            self._draw(frames, detections)
        self.frame_count += 1

        return next_frames

    @staticmethod
    def print_timing_info(stream_idx):
        LOGGER.debug(f"{'  track time:':<39}{Profiler.get_avg_millis(stream_idx, 'track'):>6.3f} ms")
        LOGGER.debug(f"{'  preprocess time:':<39}{Profiler.get_avg_millis(stream_idx, 'preproc'):>6.3f} ms")
        LOGGER.debug(f"{'  detect/flow time:':<39}{Profiler.get_avg_millis(stream_idx, 'detect'):>6.3f} ms")
        LOGGER.debug(f"{'  feature extract/kalman filter time:':<39}"
                     f"{Profiler.get_avg_millis(stream_idx, 'extract'):>6.3f} ms")
        LOGGER.debug(f"{'  association time:':<39}{Profiler.get_avg_millis(stream_idx, 'assoc'):>6.3f} ms")

    def _draw(self, frames, detections):
        for stream_idx in range(0, self.stream_num):
            visible_tracks = list(self.visible_tracks()[stream_idx])
            tracker = self.trackers[stream_idx]
            self.visualizer.render(frames[stream_idx], visible_tracks, detections[stream_idx], tracker.klt_bboxes.values(),
                                tracker.flow.prev_bg_keypoints, tracker.flow.bg_keypoints)
            cv2.putText(frames[stream_idx], f'visible: {len(visible_tracks)}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
