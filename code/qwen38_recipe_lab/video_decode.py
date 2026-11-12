"""Bounded local-file PyAV decoding for the Mac processor reference.

Two passes count actual frames then retain only selected RGB frames. Average
stream FPS drives the public index/FPS timestamp rule; this is not VFR PTS timing.
The limits and decoder backend are explicit Mac choices, not official settings.
"""
from dataclasses import dataclass
from pathlib import Path
import av
import torch
from pixels import PixelProcessor,frame_indices


@dataclass
class DecodeLimits:
    file_bytes:int=16_000_000
    source_frames:int=1024
    side:int=2048
    selected_pixels:int=16_777_216


def video_file(path,processor=None,limits=None,num_frames=None,target_fps=None):
    processor=processor or PixelProcessor();limits=limits or DecodeLimits();path=Path(path)
    if not path.is_file() or not 0<path.stat().st_size<=limits.file_bytes:raise ValueError('Local video file byte limit exceeded')
    options={'protocol_whitelist':'file,pipe'}
    with av.open(str(path),options=options) as container:
        if not container.streams.video:raise ValueError('No video stream')
        stream=container.streams.video[0];stream.codec_context.thread_count=1
        fps=float(stream.average_rate) if stream.average_rate else None
        width,height=stream.width,stream.height
        if min(width,height)<1 or max(width,height)>limits.side:raise ValueError('Video dimension limit exceeded')
        total=0
        for frame in container.decode(stream):
            total+=1
            if total>limits.source_frames:raise ValueError('Video source frame limit exceeded')
            if (frame.width,frame.height)!=(width,height):raise ValueError('Changing frame dimensions are unsupported')
    if not total:raise ValueError('Empty video')
    selected,fps=frame_indices(total,fps,processor.config,num_frames,target_fps)
    if len(selected)*height*width>limits.selected_pixels:raise ValueError('Selected RGB pixel limit exceeded')
    wanted=set(selected.tolist());decoded={}
    with av.open(str(path),options=options) as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=1
        for index,frame in enumerate(container.decode(stream)):
            if index in wanted:decoded[index]=torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2,0,1)
            if index>=int(selected[-1]):break
    if set(decoded)!=wanted:raise ValueError('Video changed between decoding passes')
    result=processor.selected_video([decoded[int(i)] for i in selected],selected,fps)
    result['video_metadata']=dict(fps=fps,total_num_frames=total,frames_indices=selected.tolist(),decoder='pyav-'+av.__version__)
    return result
