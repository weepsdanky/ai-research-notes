"""Hash-locked primary processor methods, detached from framework constructors."""
import ast,hashlib,json,math,re
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import PIL
from PIL.Image import Resampling
from torchvision.transforms.v2 import functional as tvF


def load(kind):
    files={'image':('image_processing_qwen2_vl.py','Qwen2VLImageProcessor',{'patchify'}),
        'video':('video_processing_qwen3_vl.py','Qwen3VLVideoProcessor',{'sample_frames','patchify'}),
        'backend':('image_processing_backends.py','TorchvisionBackend',{'resize','normalize','convert_to_rgb','_fuse_mean_std_and_rescale_factor','rescale_and_normalize'}),
        'color':('image_transforms.py',None,set()),
        'prompt':('processing_qwen3_vl.py','Qwen3VLProcessor',{'_calculate_timestamps','replace_image_token','replace_video_token'}),
        'packing':('processing_utils.py','ProcessorMixin',{'get_text_with_replacements','create_mm_token_type_ids'})}
    name,cls,methods=files[kind];path=Path('.cache/references')/name
    spec=next(s for s in json.loads(Path('sources.lock.json').read_text()) if s['name']==name)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
    nodes=[]
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.FunctionDef) and node.name in ('smart_resize','convert_to_rgb'):nodes.append(node)
        if isinstance(node,ast.ClassDef) and node.name==cls:
            node.bases=[];node.decorator_list=[]
            node.body=[m for m in node.body if isinstance(m,ast.FunctionDef) and m.name in methods]
            for method in node.body:method.decorator_list=[]
            nodes.append(node)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
    ns=dict(torch=torch,np=np,math=math,re=re,tvF=tvF,PIL=PIL,PILImageResampling=Resampling,
        requires_backends=lambda *args:None,ChannelDimension=SimpleNamespace(FIRST='channels_first',LAST='channels_last'),
        pil_torch_interpolation_mapping={Resampling.BICUBIC:tvF.InterpolationMode.BICUBIC},
        is_torchdynamo_compiling=lambda:False,logger=SimpleNamespace(warning_once=lambda message:None))
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns)
    if cls is not None:ns['instance']=ns[cls]()
    return ns
