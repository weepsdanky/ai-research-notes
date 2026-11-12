"""Audit released multimodal chat rendering with locked public packing methods."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path
import re
from types import SimpleNamespace

from jinja2.exceptions import TemplateError
import numpy as np
from tokenizers import Tokenizer

from mrcr_tasks import chat_template
from qwen35_quality_data import digest, load, write_json

START, END = '<|vision_start|>', '<|vision_end|>'
IMAGE, VIDEO = '<|image_pad|>', '<|video_pad|>'


def reference_class(path, name, methods):
    spec, = [entry for entry in load('sources.lock.json') if entry['name'] == path.name]
    assert digest(path) == spec['sha256']
    cls, = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == name]
    cls.bases, cls.keywords, cls.decorator_list = [], [], []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    assert {node.name for node in cls.body} == set(methods)
    namespace = dict(re=re, np=np)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]()


def audit():
    root = Path('.cache/references')
    tokenizer_file = root / 'tokenizer.json'
    lock = {entry['name']: entry for entry in load('sources.lock.json')}
    assert digest(tokenizer_file) == lock[tokenizer_file.name]['sha256']
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    config_path = root / 'config.official.json'
    assert digest(config_path) == lock['config.official.json']['sha256']
    config = load(config_path)
    for name, token in (('image', IMAGE), ('video', VIDEO), ('vision_start', START), ('vision_end', END)):
        assert tokenizer.token_to_id(token) == config[f'{name}_token_id']

    packing_path = root / 'processing_utils.py'
    prompt_path = root / 'processing_qwen3_vl.py'
    packing = reference_class(packing_path, 'ProcessorMixin',
                              ('get_text_with_replacements', 'create_mm_token_type_ids'))
    prompt = reference_class(prompt_path, 'Qwen3VLProcessor',
                             ('replace_image_token', 'replace_video_token', '_calculate_timestamps'))
    packing.all_special_multimodal_tokens = [IMAGE, VIDEO]
    for obj in (packing, prompt):
        obj.image_token, obj.video_token = IMAGE, VIDEO
        obj.vision_start_token, obj.vision_end_token = START, END
    packing.image_token_ids = [tokenizer.token_to_id(IMAGE)]
    packing.video_token_ids = [tokenizer.token_to_id(VIDEO)]
    packing.audio_token_ids = []
    prompt.image_processor = prompt.video_processor = SimpleNamespace(merge_size=2, temporal_patch_size=2)
    image_grid = {'image_grid_thw': np.array([[1, 2, 2], [1, 4, 4]])}
    video_grid = {'video_grid_thw': np.array([[2, 2, 4]]),
                  'video_metadata': [SimpleNamespace(fps=10, frames_indices=[0, 2, 4, 6])]}
    image_replacements = [prompt.replace_image_token(image_grid, i) for i in range(2)]
    video_replacement = prompt.replace_video_token(video_grid, 0)
    frames = ''.join(f'<{stamp} seconds>{START}{VIDEO * 2}{END}' for stamp in ('0.1', '0.5'))
    assert image_replacements == [IMAGE, IMAGE * 4] and video_replacement == frames

    messages = [{'role': 'system', 'content': 'Inspect the media.'},
                {'role': 'user', 'content': [{'type': 'image', 'image': 'local.png'},
                                             {'type': 'text', 'text': ' and '},
                                             {'type': 'video', 'video': 'local.mp4'},
                                             {'type': 'text', 'text': ' then '},
                                             {'type': 'image', 'image': 'other.png'}]}]
    template = chat_template()
    for item in (messages[1]['content'][0], messages[1]['content'][2]):
        try:
            template.render(messages=[{'role': 'system', 'content': [item]}, messages[1]],
                            add_generation_prompt=True, enable_thinking=False, tools=None)
        except TemplateError as exc:
            assert f'System message cannot contain {item["type"]}s.' in str(exc)
        else:
            raise AssertionError('System media was accepted')
    cases = []
    for labels in (False, True):
        rendered = template.render(messages=messages, add_generation_prompt=True,
                                          enable_thinking=False, tools=None, add_vision_id=labels)
        prefix, suffix = '<|im_start|>system\nInspect the media.<|im_end|>\n<|im_start|>user\n', (
            '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        )
        picture1, video1, picture2 = ('Picture 1: ', 'Video 1: ', 'Picture 2: ') if labels else ('', '', '')
        assert rendered == (f'{prefix}{picture1}{START}{IMAGE}{END} and {video1}{START}{VIDEO}{END}'
                            f' then {picture2}{START}{IMAGE}{END}{suffix}')
        expanded, offsets = packing.get_text_with_replacements([rendered], image_replacements,
                                                               [video_replacement])
        expected = (f'{prefix}{picture1}{START}{IMAGE}{END} and {video1}{START}{frames}{END}'
                    f' then {picture2}{START}{IMAGE * 4}{END}{suffix}')
        assert expanded == [expected]
        assert [row['type'] for row in offsets[0]] == ['image', 'video', 'image']
        for offset in offsets[0]:
            start, end = offset['span']
            assert rendered[start:end] == offset['text']
            start, end = offset['new_span']
            assert expected[start:end] == offset['replacement']
        ids = tokenizer.encode(expected, add_special_tokens=False).ids
        types, = packing.create_mm_token_type_ids([ids])
        assert Counter(types) == {0: len(ids) - 9, 1: 5, 2: 4}
        assert all(ids.count(tokenizer.token_to_id(token)) == 5 for token in (START, END))
        cases.append(dict(vision_labels=labels, rendered=rendered, expanded=expected,
                          input_ids=ids, modality_ids=types))
    paths = [tokenizer_file, config_path, packing_path, prompt_path,
             Path('.cache/long-context-sources/qwen-chat-template.jinja'),
             Path('.cache/long-context-sources/hf-chat-template-utils.py'), Path('mrcr_tasks.py'), Path(__file__)]
    return dict(scope='Released rendering and locked processor text packing with synthetic grids',
                source_sha256={str(path): digest(path) for path in paths}, cases=cases, system_media_rejections=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    assert args.out.parent.resolve() == Path('runs').resolve() and not args.out.exists()
    write_json(args.out, audit())


if __name__ == '__main__':
    main()
