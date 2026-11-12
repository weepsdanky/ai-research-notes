"""Load only audited mathematical components from the hash-locked Apache source.

No remote code execution: bootstrap downloads are separately hashed and cached.
Decorators (kernel substitutions/compilation) are removed to exercise CPU reference.
"""
import ast,hashlib,json,math,itertools
from pathlib import Path
import torch
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F

NAMES={'Qwen4ExpTextRMSNorm','Qwen4ExpTextRMSNormGated','apply_mask_to_padding_states',
       'causal_conv1d_update','causal_conv1d_fn','l2norm','torch_chunk_gated_delta_rule',
       'torch_recurrent_gated_delta_rule','Qwen4ExpTextGatedDeltaNet','rotate_half',
       'apply_rotary_pos_emb','Qwen4ExpTextQSAIndexer','Qwen4ExpTextGatedResidual',
       '_splitmix64','_build_layer_multipliers','_is_prime','_find_nth_prime_after',
       'Qwen4ExpTextNGramEmbedding','Qwen4ExpTextPLELayer','repeat_kv','eager_attention_forward',
       'Qwen4ExpVisionRotaryEmbedding','Qwen4ExpVisionMLP','Qwen4ExpVisionPatchEmbed',
       'Qwen4ExpVisionPatchMerger','apply_rotary_pos_emb_vision','Qwen4ExpVisionAttention',
       'Qwen4ExpVisionBlock','get_vision_cu_seqlens','get_vision_position_ids',
       'Qwen4ExpTextRotaryEmbedding','Qwen4ExpTextMLP','Qwen4ExpTextExperts','Qwen4ExpTextTopKRouter',
       'Qwen4ExpTextSparseMoeBlock','load_balancing_loss_func',
       'Qwen4ExpModel','Qwen4ExpTextAttention',
       '_interpolation_axis_taps_weights','get_vision_interpolation_indices_and_weights'}

def load_reference():
    root=Path(__file__).parent
    kept=[]
    for filename in ('modeling_qwen4_exp.py','vision_utils.py'):
        source=root/'.cache/references'/filename
        spec=next(x for x in json.loads((root/'sources.lock.json').read_text()) if x['name']==source.name)
        assert hashlib.sha256(source.read_bytes()).hexdigest()==spec['sha256'],'Reference drift'
        for node in ast.parse(source.read_text()).body:
            if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in NAMES:
                if node.name=='Qwen4ExpModel':
                    node.bases=[ast.Attribute(value=ast.Name(id='nn',ctx=ast.Load()),attr='Module',ctx=ast.Load())]
                    node.body=[child for child in node.body if isinstance(child,ast.FunctionDef) and child.name in ('get_vision_position_ids','get_rope_index','get_placeholder_mask')]
                for child in ast.walk(node):
                        if isinstance(child,(ast.ClassDef,ast.FunctionDef)):
                            child.decorator_list=[d for d in child.decorator_list if isinstance(d,ast.Name) and d.id in ('staticmethod','classmethod','property')]
                kept.append(node)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*kept],type_ignores=[])
    def checked(condition,message):
        if not bool(condition):raise ValueError(message)
    namespace=dict(torch=torch,nn=nn,F=F,math=math,itertools=itertools,torch_compilable_check=checked,ACT2FN={'silu':F.silu,'sigmoid':torch.sigmoid,'gelu_pytorch_tanh':lambda x:F.gelu(x,approximate='tanh')},is_torchdynamo_exporting=lambda:False,
                   GradientCheckpointingLayer=nn.Module,is_flash_attention_requested=lambda c:False,
                   ALL_ATTENTION_FUNCTIONS=SimpleNamespace(get_interface=lambda name,default:default),
                   _MASK64=(1<<64)-1,_SPLITMIX_GAMMA=0x9E3779B97F4A7C15,_SPLITMIX_M1=0xBF58476D1CE4E5B9,
                   _SPLITMIX_M2=0x94D049BB133111EB,_PRIME_1=10007)
    exec(compile(ast.fix_missing_locations(module),str(source),'exec'),namespace)
    return namespace
