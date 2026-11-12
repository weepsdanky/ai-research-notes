"""CPU RGB/frame preprocessing following the locked Qwen2/3-VL processors.

Accepts PIL/CHW images and decoded TCHW video frames. Codec/file decoding is
separate. Pixel/frame caps are explicit Mac resource choices, not official caps.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import functional as tvF


@dataclass
class PixelConfig:
    patch_size:int=16
    temporal_patch_size:int=2
    merge_size:int=2
    image_min_pixels:int=65536
    image_max_pixels:int=262144
    video_min_pixels:int=4096
    video_max_pixels:int=262144
    min_frames:int=4
    max_frames:int=64
    fps:float=2.
    mean:tuple=(.5,.5,.5)
    std:tuple=(.5,.5,.5)


def smart_size(height,width,factor,minimum,maximum,frames=None,temporal=2):
    if min(height,width,factor,minimum)<1 or maximum<minimum:raise ValueError('Invalid dimensions/pixel budget')
    if frames is not None:
        if frames<temporal:raise ValueError('Too few frames for temporal patch')
        if height<factor or width<factor:
            scale=max(factor/height,factor/width);height=int(height*scale);width=int(width*scale)
    if max(height,width)/min(height,width)>200:raise ValueError('Aspect ratio exceeds 200')
    h,w=round(height/factor)*factor,round(width/factor)*factor
    count=1 if frames is None else round(frames/temporal)*temporal
    volume=height*width*(1 if frames is None else frames)
    if count*h*w>maximum:
        scale=math.sqrt(volume/maximum);h=max(factor,math.floor(height/scale/factor)*factor);w=max(factor,math.floor(width/scale/factor)*factor)
    elif count*h*w<minimum:
        scale=math.sqrt(minimum/volume);h=math.ceil(height*scale/factor)*factor;w=math.ceil(width*scale/factor)*factor
    return h,w


def patchify(frames,c):
    t,channels,h,w=frames.shape
    if pad:=-t%c.temporal_patch_size:frames=torch.cat((frames,frames[-1:].expand(pad,-1,-1,-1)));t+=pad
    gt,gh,gw=t//c.temporal_patch_size,h//c.patch_size,w//c.patch_size
    patches=frames.reshape(1,gt,c.temporal_patch_size,channels,gh//c.merge_size,c.merge_size,c.patch_size,gw//c.merge_size,c.merge_size,c.patch_size)
    patches=patches.permute(0,1,4,7,5,8,3,2,6,9).reshape(gt*gh*gw,channels*c.temporal_patch_size*c.patch_size**2)
    return patches,torch.tensor([[gt,gh,gw]],dtype=torch.long)


def frame_indices(total,source_fps,c,num_frames=None,target_fps=None):
    if total<1 or (num_frames is not None and target_fps is not None):raise ValueError('Invalid frame sampling request')
    fps=source_fps if source_fps is not None else 24.
    if fps<=0:raise ValueError('Source fps must be positive')
    if num_frames is None:
        target=c.fps if target_fps is None else target_fps
        num_frames=min(max(int(total/fps*target),c.min_frames),c.max_frames,total)
    if not 0<num_frames<=c.max_frames:raise ValueError('Explicit frame count exceeds declared Mac budget')
    return np.linspace(0,total-1,num_frames).round().astype(int),fps


def timestamps(indices,fps,merge=2):
    indices=list(map(int,indices))
    indices.extend([indices[-1]]*(-len(indices)%merge))
    values=[i/fps for i in indices]
    return [(values[i]+values[i+merge-1])/2 for i in range(0,len(values),merge)]


class PixelProcessor:
    def __init__(self,config=None):self.config=config or PixelConfig()
    def rgb(self,image):
        if isinstance(image,Image.Image):
            if 'A' in image.getbands() or image.info.get('transparency') is not None:
                rgba=image.convert('RGBA');background=Image.new('RGBA',rgba.size,(255,255,255,255))
                image=Image.alpha_composite(background,rgba)
            return tvF.pil_to_tensor(image.convert('RGB'))
        if isinstance(image,np.ndarray):
            image=torch.from_numpy(image.copy())
            if image.ndim==3:image=image.permute(2,0,1)
        image=image.to('cpu')
        if image.ndim==2:image=image[None]
        if image.ndim!=3:raise ValueError('Expected CHW RGB tensor')
        if image.shape[0]==1:image=image.expand(3,-1,-1)
        elif image.shape[0]==4:
            original=image.dtype;unit=tvF.to_dtype(image,torch.float32,scale=True)
            image=tvF.to_dtype(unit[:3]*unit[3:]+(1-unit[3:]),original,scale=True)
        if image.shape[0]!=3:raise ValueError('Expected 1, 3 or 4 channels')
        return image
    def normalized(self,frames,size):
        resized=tvF.resize(frames,list(size),interpolation=tvF.InterpolationMode.BICUBIC,antialias=True)
        c=self.config
        mean=torch.tensor(c.mean)*255.;std=torch.tensor(c.std)*255.
        return tvF.normalize(resized.to(torch.float32),mean,std)
    def image(self,image):
        c=self.config;image=self.rgb(image)
        size=smart_size(*image.shape[-2:],c.patch_size*c.merge_size,c.image_min_pixels,c.image_max_pixels)
        values,grid=patchify(self.normalized(image[None],size),c)
        return dict(pixel_values=values,image_grid_thw=grid)
    def video(self,frames,source_fps=None,num_frames=None,target_fps=None):
        c=self.config
        if isinstance(frames,np.ndarray):frames=torch.from_numpy(frames.copy()).permute(0,3,1,2)
        if frames.ndim!=4:raise ValueError('Expected decoded TCHW video')
        selected,fps=frame_indices(len(frames),source_fps,c,num_frames,target_fps)
        return self.selected_video(frames[selected.tolist()],selected,fps)
    def selected_video(self,frames,selected,fps):
        """Normalize already selected frames while retaining original timestamps."""
        c=self.config
        if not len(frames) or len(frames)!=len(selected) or not math.isfinite(fps) or fps<=0:raise ValueError('Invalid selected video metadata')
        frames=torch.stack([self.rgb(frame) for frame in frames])
        size=smart_size(*frames.shape[-2:],c.patch_size*c.merge_size,c.video_min_pixels,c.video_max_pixels,len(frames),c.temporal_patch_size)
        values,grid=patchify(self.normalized(frames,size),c)
        return dict(pixel_values_videos=values,video_grid_thw=grid,frame_indices=list(map(int,selected)),timestamps=timestamps(selected,fps,c.temporal_patch_size))
