"""Real lossless container decoding and locked HF replacement-method oracles."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import av
import torch
from PIL import Image
from pixels import PixelConfig,PixelProcessor
from processor import Processor,model_inputs,IMAGE,VIDEO,START,END
from processor_reference import load
from video_decode import video_file,DecodeLimits

torch.set_num_threads(1)


def fixture(directory):
    frames=(torch.arange(9*3*12*16).reshape(9,3,12,16)%256).byte();path=Path(directory)/'lossless.mov'
    with av.open(str(path),'w') as container:
        stream=container.add_stream('png',rate=24);stream.width=16;stream.height=12;stream.pix_fmt='rgb24';stream.codec_context.thread_count=1
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame.permute(1,2,0).numpy(),format='rgb24')):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    return path,frames


def pixels():
    return PixelProcessor(PixelConfig(patch_size=2,temporal_patch_size=2,merge_size=2,image_min_pixels=64,
        image_max_pixels=1024,video_min_pixels=64,video_max_pixels=1024,max_frames=8))


class ProcessorTests(unittest.TestCase):
    def test_real_file_matches_decoded_frame_oracle_and_limits(self):
        with tempfile.TemporaryDirectory(dir='runs',prefix='codec-unit-') as root:
            path,frames=fixture(root);p=pixels();actual=video_file(path,p,num_frames=3);expected=p.video(frames,24,num_frames=3)
            for name in ('pixel_values_videos','video_grid_thw'):torch.testing.assert_close(actual[name],expected[name],atol=0,rtol=0)
            for name in ('frame_indices','timestamps'):self.assertEqual(actual[name],expected[name])
            self.assertEqual(actual['video_metadata']['fps'],24);self.assertEqual(actual['video_metadata']['total_num_frames'],9)
            for cap in (DecodeLimits(file_bytes=1),DecodeLimits(source_frames=2),DecodeLimits(side=8),DecodeLimits(selected_pixels=2)):
                with self.assertRaises(ValueError):video_file(path,p,cap,num_frames=3)

    def test_locked_hf_expansion_token_ids_types_and_padding(self):
        with tempfile.TemporaryDirectory(dir='runs',prefix='packing-unit-') as root:
            path,frames=fixture(root);p=pixels();processor=Processor(p);image=Image.fromarray(frames[0].permute(1,2,0).numpy())
            image_path=Path(root)/'image.png';image.save(image_path)
            text=[f'Look {START}{IMAGE}{END}, then {VIDEO}.','Short text']
            iv=p.image(image);vv=p.video(frames,24,num_frames=3);ref=load('prompt')['instance']
            ref.image_processor=ref.video_processor=p.config;ref.image_token=IMAGE;ref.video_token=VIDEO;ref.vision_start_token=START;ref.vision_end_token=END
            ir=ref.replace_image_token(iv,0)
            vr=ref.replace_video_token(dict(video_grid_thw=vv['video_grid_thw'],video_metadata=[SimpleNamespace(fps=24,frames_indices=vv['frame_indices'])]),0)
            base=load('packing')['instance'];base.all_special_multimodal_tokens=[IMAGE,VIDEO];base.image_token=IMAGE;base.video_token=VIDEO
            base.image_token_ids=[processor.image_id];base.video_token_ids=[processor.video_id];base.audio_token_ids=[]
            expanded,_=base.get_text_with_replacements(text.copy(),[ir],[vr])
            for side in ('left','right'):
                batch=processor(text,[image if side=='left' else image_path],[path],num_frames=3,padding_side=side)
                self.assertTrue(torch.all(batch['input_ids'][batch['attention_mask']==0]==processor.pad_id))
                self.assertEqual(batch['expanded_text'],expanded)
                for i,s in enumerate(expanded):
                    self.assertEqual(batch['input_ids'][i,batch['attention_mask'][i].bool()].tolist(),processor.tokenizer.encode(s,add_special_tokens=False).ids)
                self.assertEqual(batch['mm_token_type_ids'].tolist(),base.create_mm_token_type_ids(batch['input_ids'].tolist()))
                torch.testing.assert_close(batch['pixel_values'],iv['pixel_values'],atol=0,rtol=0)
                torch.testing.assert_close(batch['pixel_values_videos'],vv['pixel_values_videos'],atol=0,rtol=0)
            with self.assertRaises(ValueError):processor(IMAGE)
            with self.assertRaises(ValueError):Processor(p,max_tokens=1)('hello world')

    def test_encoded_media_to_visual_encoder_text_logits_and_gradients(self):
        from model import Model
        from test_recipe import tiny
        from vision import VisionConfig,VisionModel
        from multimodal import MultimodalModel
        torch.manual_seed(71);device=os.environ.get('LAB_TEST_DEVICE','cpu')
        with tempfile.TemporaryDirectory(dir='runs',prefix='connected-media-') as root:
            path,frames=fixture(root);p=pixels();image=Image.fromarray(frames[0].permute(1,2,0).numpy())
            batch=Processor(p)(f'{START}{IMAGE}{END} and {VIDEO}',[image],[path],num_frames=3)
            text=Model(tiny(vocab_size=248320,eos=248044,width=16,rank=2))
            visual=VisionModel(VisionConfig(hidden_size=16,intermediate_size=32,num_heads=2,depth=1,patch_size=2,out_hidden_size=16))
            model=MultimodalModel(text,visual).to(device)
            kwargs={k:v.to(device) for k,v in model_inputs(batch).items()}
            logits,positions,deltas=model(**kwargs,stage='pretrain')
            self.assertTrue(torch.isfinite(logits).all());self.assertEqual(positions.shape,(3,*batch['input_ids'].shape))
            logits[...,7].square().mean().backward()
            grads=[p.grad for p in visual.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in grads));self.assertGreater(sum(float(g.abs().sum()) for g in grads),0)


if __name__=='__main__':unittest.main()
