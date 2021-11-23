from pathlib import Path
from enum import Enum
from collections import deque
from urllib.parse import urlparse
import numpy as np
import subprocess
import threading
import logging
import cv2
import PyNvCodec as nvc


LOGGER = logging.getLogger(__name__)


class Protocol(Enum):
    IMAGE = 0
    VIDEO = 1
    CSI   = 2
    V4L2  = 3
    RTSP  = 4
    HTTP  = 5


class VideoIO:
    def __init__(self, size, input_uri,
                 output_uri=None,
                 output_rtsp=None,
                 resolution=(1920, 1080),
                 frame_rate=30,
                 buffer_size=10,
                 proc_fps=30):
        """Class for video capturing and output saving.
        Encoding, decoding, and scaling can be accelerated using the GStreamer backend.

        Parameters
        ----------
        size : tuple
            Width and height of each frame to output.
        input_uri : str
            URI to input stream. It could be image sequence (e.g. '%06d.jpg'), video file (e.g. 'file.mp4'),
            MIPI CSI camera (e.g. 'csi://0'), USB/V4L2 camera (e.g. '/dev/video0'),
            RTSP stream (e.g. 'rtsp://<user>:<password>@<ip>:<port>/<path>'),
            or HTTP live stream (e.g. 'http://<user>:<password>@<ip>:<port>/<path>')
        output_uri : str, optionals
            URI to an output video file.
        output_rtsp : str, optional
            RTSP Server URI to output video
        resolution : tuple, optional
            Original resolution of the input source.
            Useful to set a certain capture mode of a USB/CSI camera.
        frame_rate : int, optional
            Frame rate of the input source.
            Required if frame rate cannot be deduced, e.g. image sequence and/or RTSP.
            Useful to set a certain capture mode of a USB/CSI camera.
        buffer_size : int, optional
            Number of frames to buffer.
            For live sources, a larger buffer drops less frames but increases latency.
        proc_fps : int, optional
            Estimated processing speed that may limit the capture interval `cap_dt`.
            This depends on hardware and processing complexity.
        """
        self.size = size
        self.input_uri = input_uri
        self.output_uri = output_uri
        self.output_rtsp = output_rtsp
        self.resolution = resolution
        assert frame_rate > 0
        self.frame_rate = frame_rate
        assert buffer_size >= 1
        self.buffer_size = buffer_size
        assert proc_fps > 0
        self.proc_fps = proc_fps

        self.protocol = self._parse_uri(self.input_uri)
        self.is_live = self.protocol != Protocol.IMAGE and self.protocol != Protocol.VIDEO

        self.cc_ctx = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG)
        self.rawSurface = None
        self.nvDec = None
        self.nvDwn = nvc.PySurfaceDownloader(self.size[0], self.size[1], nvc.PixelFormat.BGR, 0)
        self.nvCvtNv12ToYuv420 = nvc.PySurfaceConverter(self.size[0], self.size[1], nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420, 0)
        self.nvCvtYuv420ToBgr = nvc.PySurfaceConverter(self.size[0], self.size[1], nvc.PixelFormat.YUV420, nvc.PixelFormat.BGR, 0)

        # TODO: obtain real values from the stream for further usage
        width = self.size[0]
        height = self.size[1]
        self.cap_fps =  self.frame_rate

        self.do_resize = (width, height) != self.size
        if self.cap_fps == 0:
            self.cap_fps = self.frame_rate # fallback to config if unknown
        LOGGER.info('%dx%d stream @ %d FPS', width, height, self.cap_fps)

        self.nvEnc = None
        if self.output_uri is not None:
            self.output_file = open(self.output_uri, "wb")
            self.nvEnc = nvc.PyNvEncoder({'preset': 'P1', 'tuning_info' : 'high_quality', 'codec': 'h264', 
                                        'profile' : 'main', 's': '1920x1080', 'bitrate' : '10M'}, 0)
        
        if self.output_rtsp is not None:
            write_to_rtsp_command = ['ffmpeg',
                '-re',
                '-f', 'rawvideo',
                '-s', f'{self.size[0]}x{self.size[1]}',
                '-pixel_format', 'bgr24',
                '-r', f'{self.cap_fps}',
                '-i', '-',
                '-pix_fmt', 'yuv420p',
                '-c:v', 'h264_nvenc',
                '-preset:v', 'llhq',
                '-profile:v', 'high',                
                '-bufsize', '64M',
                '-maxrate', '4M',
                '-rtsp_transport', 'tcp',
                '-f', 'rtsp',
                #'-muxdelay', '0.1',
                self.output_rtsp
            ]
            self.rtsp_writer_process = subprocess.Popen(write_to_rtsp_command, stdin=subprocess.PIPE)
            

    @property
    def cap_dt(self):
        # limit capture interval at processing latency for live sources
        return 1 / min(self.cap_fps, self.proc_fps) if self.is_live else 1 / self.cap_fps

    def read(self):
        """Reads the next video frame.

        Returns
        -------
        ndarray
            Returns None if there are no more frames.
        """
        if self.nvDec is None:
            self.nvDec = nvc.PyNvDecoder(self.input_uri, 0, {'rtsp_transport': 'tcp', 
            "max_delay": "5000000", "bufsize": "300000k"})
           
        self.rawSurface = self.nvDec.DecodeSingleSurface()
        if (self.rawSurface.Empty()):
            return None
    
        surface_yuv420 = self.nvCvtNv12ToYuv420.Execute(self.rawSurface, self.cc_ctx)
        if surface_yuv420.Empty():
            return None

        surface_rgb = self.nvCvtYuv420ToBgr.Execute(surface_yuv420, self.cc_ctx)
        if surface_rgb.Empty():
            return None

        frame = np.ndarray(shape=(self.size[1], self.size[0], 3), dtype=np.uint8)
        success = self.nvDwn.DownloadSingleSurface(surface_rgb, frame)
        if not (success):
            return None

        return frame

    def write_frame(self, frame):
        self.writer.stdin.write(frame.tobytes())

    def write(self, frame):
        """Writes the next video frame."""
        encFrame = np.ndarray(shape=(0), dtype=np.uint8)

        success = self.nvEnc.EncodeSingleSurface(self.rawSurface, encFrame)
        if success:
            frameByteArray = bytearray(encFrame)
            self.output_file.write(frameByteArray)

    def write_rtsp(self, frame):
        """Writes the next video frame to rtsp server."""
        self.rtsp_writer_process.stdin.write(frame.tobytes())

    def release(self):
        """Cleans up input and output sources."""
        if (self.nvEnc is not None):
            encFrame = np.ndarray(shape=(0), dtype=np.uint8)
            while True:
                success = self.nvEnc.FlushSinglePacket(encFrame)
                if(success):
                    encByteArray = bytearray(encFrame)
                    self.output_file.write(encByteArray)
                else:
                    break

        if hasattr(self, 'rtsp_writer_process'):
            self.rtsp_writer_process.stdin.close()
            self.rtsp_writer_process.wait()

    def _gst_cap_pipeline(self):
        gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
        if 'nvvidconv' in gst_elements and self.protocol != Protocol.V4L2:
            # format conversion for hardware decoder
            cvt_pipeline = (
                'nvvidconv interpolation-method=5 ! '
                'video/x-raw, width=%d, height=%d, format=BGRx !'
                'videoconvert ! appsink sync=false'
                % self.size
            )
        else:
            cvt_pipeline = (
                'videoscale ! '
                'video/x-raw, width=%d, height=%d !'
                'videoconvert ! appsink sync=false'
                % self.size
            )

        if self.protocol == Protocol.IMAGE:
            pipeline = (
                'multifilesrc location=%s index=1 caps="image/%s,framerate=%d/1" ! decodebin ! '
                % (
                    self.input_uri,
                    self._img_format(self.input_uri),
                    self.frame_rate
                )
            )
        elif self.protocol == Protocol.VIDEO:
            pipeline = 'filesrc location=%s ! decodebin ! ' % self.input_uri
        elif self.protocol == Protocol.CSI:
            if 'nvarguscamerasrc' in gst_elements:
                pipeline = (
                    'nvarguscamerasrc sensor_id=%s ! '
                    'video/x-raw(memory:NVMM), width=%d, height=%d, '
                    'format=NV12, framerate=%d/1 ! '
                    % (
                        self.input_uri[6:],
                        *self.resolution,
                        self.frame_rate
                    )
                )
            else:
                raise RuntimeError('GStreamer CSI plugin not found')
        elif self.protocol == Protocol.V4L2:
            if 'v4l2src' in gst_elements:
                pipeline = (
                    'v4l2src device=%s ! '
                    'video/x-raw, width=%d, height=%d, '
                    'format=YUY2, framerate=%d/1 ! '
                    % (
                        self.input_uri,
                        *self.resolution,
                        self.frame_rate
                    )
                )
            else:
                raise RuntimeError('GStreamer V4L2 plugin not found')
        elif self.protocol == Protocol.RTSP:
            pipeline = (
                'rtspsrc location=%s latency=0 ! '
                'capsfilter caps=application/x-rtp,media=video ! decodebin ! ' % self.input_uri
            )
        elif self.protocol == Protocol.HTTP:
            pipeline = 'souphttpsrc location=%s is-live=true ! decodebin ! ' % self.input_uri
        return pipeline + cvt_pipeline

    def _gst_write_pipeline(self):
        gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
        # use hardware encoder if found
        if 'omxh264enc' in gst_elements:
            h264_encoder = 'omxh264enc preset-level=2'
        elif 'x264enc' in gst_elements:
            h264_encoder = 'x264enc pass=4'
        else:
            raise RuntimeError('GStreamer H.264 encoder not found')
        pipeline = (
            'appsrc ! autovideoconvert ! %s ! qtmux ! filesink location=%s '
            % (
                h264_encoder,
                self.output_uri
            )
        )
        return pipeline

    @staticmethod
    def _parse_uri(uri):
        result = urlparse(uri)
        if result.scheme == 'csi':
            protocol = Protocol.CSI
        elif result.scheme == 'rtsp':
            protocol = Protocol.RTSP
        elif result.scheme == 'http':
            protocol = Protocol.HTTP
        else:
            if '/dev/video' in result.path:
                protocol = Protocol.V4L2
            elif '%' in result.path:
                protocol = Protocol.IMAGE
            else:
                protocol = Protocol.VIDEO
        return protocol

    @staticmethod
    def _img_format(uri):
        img_format = Path(uri).suffix[1:]
        return 'jpeg' if img_format == 'jpg' else img_format
