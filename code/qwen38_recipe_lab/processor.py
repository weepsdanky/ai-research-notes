"""Bounded text/image/video packing following locked Qwen3-VL replacements.

Takes prepared text placeholders, not a full HF chat/conversation API. Image
markers retain their caller-provided vision wrappers; video markers expand to
per-frame timestamps and wrappers. No truncation or production training claim.
"""
import re
from pathlib import Path
import torch
from PIL import Image
from tokenizers import Tokenizer
from pixels import PixelProcessor
from video_decode import video_file

IMAGE='<|image_pad|>';VIDEO='<|video_pad|>';START='<|vision_start|>';END='<|vision_end|>'


class Processor:
    def __init__(self,pixels=None,tokenizer=None,max_tokens=512):
        self.pixels=pixels or PixelProcessor();self.tokenizer=tokenizer or Tokenizer.from_file('.cache/references/tokenizer.json')
        self.max_tokens=max_tokens
        self.image_id=self.tokenizer.token_to_id(IMAGE);self.video_id=self.tokenizer.token_to_id(VIDEO)
        self.pad_id=self.tokenizer.token_to_id('<|endoftext|>')
        if self.image_id is None or self.video_id is None:raise ValueError('Vision tokenizer tokens required')

    def __call__(self,text,images=None,videos=None,num_frames=None,target_fps=None,padding_side='right'):
        text=[text] if isinstance(text,str) else list(text);images=[] if images is None else list(images);videos=[] if videos is None else list(videos)
        if not 0<len(text)<=2 or not all(isinstance(t,str) and len(t.encode())<=1_000_000 for t in text) or padding_side not in ('left','right'):raise ValueError('Bounded text batch and padding side required')
        if len(images)>8 or len(videos)>4:raise ValueError('Mac media item limit exceeded')
        if sum(t.count(IMAGE) for t in text)!=len(images) or sum(t.count(VIDEO) for t in text)!=len(videos):raise ValueError('Placeholder/media count mismatch')
        image_inputs=[];video_inputs=[];replacements={IMAGE:[],VIDEO:[]};merge=self.pixels.config.merge_size**2
        for image in images:
            if isinstance(image,(str,Path)):
                path=Path(image)
                if not path.is_file() or path.stat().st_size>16_000_000:raise ValueError('Image file byte limit exceeded')
                with Image.open(path) as source:
                    if max(source.size)>2048:raise ValueError('Image dimension limit exceeded')
                    value=self.pixels.image(source)
            else:value=self.pixels.image(image)
            image_inputs.append(value);replacements[IMAGE].append(IMAGE*(int(value['image_grid_thw'].prod())//merge))
        for video in videos:
            if isinstance(video,(str,Path)):value=video_file(video,self.pixels,num_frames=num_frames,target_fps=target_fps)
            elif isinstance(video,dict):value=self.pixels.video(video['frames'],video.get('source_fps'),num_frames,target_fps)
            else:value=self.pixels.video(video,num_frames=num_frames,target_fps=target_fps)
            video_inputs.append(value);grid=value['video_grid_thw'][0];count=int(grid[1:].prod())//merge
            replacements[VIDEO].append(''.join(f'<{stamp:.1f} seconds>'+START+VIDEO*count+END for stamp in value['timestamps']))
        iters={token:iter(values) for token,values in replacements.items()}
        expanded=[re.sub(re.escape(IMAGE)+'|'+re.escape(VIDEO),lambda m:next(iters[m.group()]),t) for t in text]
        rows=[self.tokenizer.encode(t,add_special_tokens=False).ids for t in expanded]
        if not all(0<len(row)<=self.max_tokens for row in rows):raise ValueError('Prepared input exceeds context; no truncation')
        length=max(map(len,rows));ids=torch.full((len(rows),length),self.pad_id,dtype=torch.long);mask=torch.zeros_like(ids)
        for i,row in enumerate(rows):
            start=0 if padding_side=='right' else length-len(row);ids[i,start:start+len(row)]=torch.tensor(row);mask[i,start:start+len(row)]=1
        types=torch.where(ids==self.image_id,1,torch.where(ids==self.video_id,2,0))
        result=dict(input_ids=ids,attention_mask=mask,mm_token_type_ids=types,expanded_text=expanded,
            video_timestamps=[v['timestamps'] for v in video_inputs])
        for values,pixel_key,grid_key in [(image_inputs,'pixel_values','image_grid_thw'),(video_inputs,'pixel_values_videos','video_grid_thw')]:
            if values:result.update({pixel_key:torch.cat([v[pixel_key] for v in values]),grid_key:torch.cat([v[grid_key] for v in values])})
        return result


def model_inputs(batch):
    names={'input_ids':'ids','mm_token_type_ids':'types','attention_mask':'mask','pixel_values':'image_patches',
           'image_grid_thw':'image_grid','pixel_values_videos':'video_patches','video_grid_thw':'video_grid'}
    return {target:batch[source] for source,target in names.items() if source in batch}
